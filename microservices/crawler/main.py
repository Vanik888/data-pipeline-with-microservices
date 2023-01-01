import argparse
import asyncio
import datetime as dt
import os
import sys
import logging
import urllib.parse
from typing import Dict, Any, List

from pathlib import Path
from aiohttp import ClientSession
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel

load_dotenv(find_dotenv())

BASE_URL = os.environ.get("BASE_URL", "https://www.urparts.com/")
BASE_HREF = os.environ.get("BASE_HREF", "index.cfm/page/catalogue")
MODELS_LIMIT = os.environ.get("MODELS_LIMIT", 3)


logger = logging.getLogger(__name__)


def init_base_logging(path: str = None):
    path = path if path else sys.stdout
    logging.basicConfig(
        level=logging.INFO,
        stream=path,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class Row(BaseModel):
    manufacturer: str
    category: str
    model: str
    part: str
    part_category: str
    created_at: str = dt.datetime.now()


class Base:
    _selector: str = None

    def __init__(
        self,
        base_url: str,
        href: str,
        parents: List[str] = None,
        parser_type: str = "html.parser",
    ):
        self.href = href
        self.base_url = base_url
        self.parents = parents
        self.parser_type = parser_type
        self._parser = None
        self._content = None

    async def open(self, session, **kwargs):
        logger.debug(f"opening the page: {self.__class__.__name__}")
        r = await session.request(method="GET", url=self.url, **kwargs)
        r.raise_for_status()
        self._content = await r.text()
        logger.debug(f"parsed the page: {self.__class__.__name__}")
        return self

    @property
    def url(self):
        return urllib.parse.urljoin(self.base_url, self.href)

    @property
    def parser(self):
        if not self._parser:
            self._parser = BeautifulSoup(self._content, features=self.parser_type)
        return self._parser

    def get_items(self) -> Dict[str, Any]:
        logger.debug(f"parsing the {self.__class__.__name__} in: {self.url}")
        return {
            tag.text.strip(): tag.get("href")
            for tag in self.parser.select(self._selector)
        }

    def count_items(self):
        return len(self.parser.select(self._selector))


class MainPage(Base):
    _selector = "#content ul li a"

    def get_manufacturers(self):
        return [
            ManufacturersPage(
                self.base_url,
                href,
                parents=[
                    name,
                ],
            )
            for name, href in self.get_items().items()
        ]


class ManufacturersPage(Base):
    _selector = "#content ul li a"

    def get_categories(self):
        return [
            CategoriesPage(self.base_url, href, parents=[*self.parents, name])
            for name, href in self.get_items().items()
        ]


class CategoriesPage(Base):
    _selector = "#content div.c_container.allmodels ul > li > a"

    def get_models(self):
        return [
            ModelsPage(self.base_url, href, parents=[*self.parents, name])
            for name, href in self.get_items().items()
        ]


class ModelsPage(Base):
    _selector = "#content div.c_container.allparts ul > li > a"

    def get_parts(self) -> List[Row]:
        for text in self.get_items().keys():
            delimiter_inx = text.find("-")
            yield Row(
                manufacturer=self.parents[0],
                category=self.parents[1],
                model=self.parents[2],
                part=text[:delimiter_inx].strip(),
                part_category=text[delimiter_inx + 1 :].strip(),
            )


async def execute_in_chunks(chunk_size: int, tasks: List):
    for j, i in enumerate(range(0, len(tasks), chunk_size), start=1):
        chunk = tasks[i : chunk_size + i]
        logger.info(f"open chunk #{j} with {len(chunk)} tasks")
        await asyncio.gather(*chunk)
        logger.info(f"opened chunk #{j} with {len(chunk)} tasks")


async def open_child_pages(
    parent_pages: List[Base], session: ClientSession, chunk_size: int = 10
):
    tasks = [pp.open(session=session) for pp in parent_pages]
    try:
        page_type = parent_pages[0].__class__.__name__
    except IndexError:
        page_type = None

    logger.info(f"open {len(tasks)} {page_type} pages asynchronously")
    await execute_in_chunks(tasks=tasks, chunk_size=chunk_size)
    logger.info(f"finished to open {len(tasks)} {page_type} pages asynchronously")


async def run_crawling_tasks(
    base_url: str, href: str, chunk_size: int, models_limit: int
):

    async with ClientSession() as session:
        mp = MainPage(base_url=base_url, href=href, parents=[])
        await open_child_pages(
            parent_pages=[
                mp,
            ],
            session=session,
            chunk_size=chunk_size,
        )

        man_pages = mp.get_manufacturers()
        await open_child_pages(
            parent_pages=man_pages, session=session, chunk_size=chunk_size
        )

        category_pages = [
            cat_page for man_page in man_pages for cat_page in man_page.get_categories()
        ]
        await open_child_pages(
            parent_pages=category_pages, session=session, chunk_size=chunk_size
        )

        model_pages = [
            model_page
            for category_page in category_pages
            for model_page in category_page.get_models()
        ]
        if models_limit:
            model_pages = model_pages[:models_limit]
        await open_child_pages(
            parent_pages=model_pages, session=session, chunk_size=chunk_size
        )

        parts = [part for model_page in model_pages for part in model_page.get_parts()]

        logger.info(f"extracted: {len(parts)} parts")

        df = pd.DataFrame([p.dict() for p in parts])
        data_dir = os.path.join(Path(__file__).parent.absolute(), "data")
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(
            data_dir, f"result-{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')}.csv"
        )
        df.to_csv(filename, index=False)
        logger.info(f"stored results in {filename}")


if __name__ == "__main__":
    init_base_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default=BASE_URL,
        help="The URL of the main page",
    )
    parser.add_argument(
        "--href",
        type=str,
        default=BASE_HREF,
        help="Href of the manufacturers",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=10,
        help="The chunk size of the async tasks",
    )
    parser.add_argument(
        "-l",
        "--models-limit",
        type=int,
        default=MODELS_LIMIT,
        help="Limit the amount of the models parsed",
    )
    args = parser.parse_args()
    started = dt.datetime.now()
    logger.info(f"started with args: {args}")
    asyncio.run(
        run_crawling_tasks(
            base_url=args.url,
            href=args.href,
            chunk_size=args.chunk_size,
            models_limit=args.models_limit,
        )
    )
    finished = dt.datetime.now()
    logger.info(f"done after: {(finished-started).total_seconds()} seconds")
