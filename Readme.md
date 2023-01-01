
connect to db

```shell
docker exec -it currency-downloader-dwh /bin/bash
psql -U postgres
```

remove the volumes
```shell
docker-compose down --volumes
```

