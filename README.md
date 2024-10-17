

## Build image
```
docker build --rm -t graphrag_bim .
```

## Start Container

``` shell
docker-compose up
```

## NEO4j Commands

All nodes and relationships.
```
MATCH (n) DETACH DELETE n
```

All indexes and constraints.
```
CALL apoc.schema.assert({},{},true) YIELD label, key RETURN *
```