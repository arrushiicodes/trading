_cache = {}

async def cache_get(key):
    return _cache.get(key)

async def cache_set(key, value, ttl=300):
    _cache[key] = value
    return True

async def cache_delete(key):
    _cache.pop(key, None)
    return True
