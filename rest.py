"""Defines Web API"""

import contextlib, typing

from fastapi import FastAPI, HTTPException
from fastapi.types import DecoratedCallable

from data import MinerDB, MinerDB_V1
from error import MinerException

T = typing.TypeVar("T")
P = typing.ParamSpec("P")
DCWrapper = typing.Callable[[DecoratedCallable], DecoratedCallable]


def isrestmethod(fn: typing.Callable):
    """
    Whether callable is a registered REST method.
    """

    return hasattr(fn, "__is_restmethod__")


@contextlib.contextmanager
def handle_db_action():
    """
    Opens a context to handle `MinerException`s
    and raise them as `HTTPException`s.
    """

    try:
        yield
    except MinerException as exc:
        raise http_from_miner(exc) from exc


def http_from_miner(exc: MinerException):
    """
    Create `HTTPException` from `MinerException`.
    """

    return HTTPException(exc.status_code or 500, exc.message)


def restmethod(mname: str, path: str, **kwds) -> DCWrapper:
    """
    Marks/registers a callable as a method used
    for some REST call.
    """

    def inner(fn):
        setattr(fn, "__is_restmethod__", 1)
        setattr(fn, "__restmethod_meta__", (mname, path, kwds))
        return classmethod(fn)

    return inner


class MinerAPIMeta(typing._ProtocolMeta):
    app_cls: type[FastAPI]
    dat_cls: type[MinerDB]

    @property
    def app(cls) -> FastAPI:
        """API driver."""

        return cls._app

    @property
    def dat(cls) -> MinerDB:
        """MinerDB Interface Object."""

        return cls._dat

    @property
    def prefix(cls) -> str:
        """
        URI component which proceed subsequent
        endpoints.
        """

        return cls._prefix

    def __new__(
        mcls,
        name,
        bases,
        namespace,
        *,
        debug=False,
        prefix=None):

        cls = super().__new__(mcls, name, bases, namespace)

        # Apply class attributes.
        if cls.app_cls:
            cls._app = cls.app_cls(debug=debug)
        if cls.dat_cls:
            cls._dat = cls.dat_cls(debug=debug)
            cls._dat.init()
        cls._prefix = prefix or "/"

        for name in dir(cls):
            attr = getattr(cls, name, None)
            if attr is None or not isrestmethod(attr):
                continue

            mname, path, kwds = getattr(attr, "__restmethod_meta__")
            method = getattr(cls, mname.lower())
            setattr(cls, name, method(path, **kwds)(attr))

        return cls

    def join(self, uri):
        """
        Joins the path with API path components.
        """

        ret = "/".join((self.prefix, uri))
        while "//" in ret:
            ret = ret.replace("//", "/")

        return ret

    def delete(self, path: str, **kwds):
        """
        Wrap a callable as a DELETE function.
        """

        return self.app.delete(self.join(path), **kwds)

    def get(self, path: str, **kwds):
        """
        Wrap a callable as a GET function.
        """

        return self.app.get(self.join(path), **kwds)

    def head(self, path: str, **kwds):
        """
        Wrap a callable as a HEAD function.
        """

        return self.app.head(self.join(path), **kwds)

    def options(self, path: str, **kwds):
        """
        Wrap a callable as a OPTIONS function.
        """

        return self.app.options(self.join(path), **kwds)

    def patch(self, path: str, **kwds):
        """
        Wrap a callable as a PATCH function.
        """

        return self.app.patch(self.join(path), **kwds)

    def post(self, path: str, **kwds):
        """
        Wrap a callable as a POST function.
        """

        return self.app.post(self.join(path), **kwds)

    def put(self, path: str, **kwds):
        """
        Wrap a callable as a PUT function.
        """

        return self.app.put(self.join(path), **kwds)

    def trace(self, path: str, **kwds):
        """
        Wrap a callable as a TRACE function.
        """

        return self.app.trace(self.join(path), **kwds)


class MinerAPI(typing.Protocol, metaclass=MinerAPIMeta):
    """MinerDB API Interface Object."""

    app_cls = FastAPI
    dat_cls = None

    @property
    def app(self) -> FastAPI:
        return self._app

    @property
    def dat(self) -> MinerDB:
        return self._dat


class MinerAPI_V1(MinerAPI, debug=True, prefix="/v1/"):
    """MinerAPI version: 1"""

    dat_cls = MinerDB_V1

    @restmethod("GET", "/")
    async def read_root(cls):
        """Root of MinerAPI."""

        return {"status": "OK"}

    @restmethod("GET", "/aliases")
    def read_aliases_all(cls):
        """
        Returns all available resource aliases.
        """

        return cls.dat.find_aliases()

    @restmethod("GET", "/aliases/{name}")
    def read_aliases_one(cls, name: str):
        """
        Returns all aliases that match a name.
        """

        return cls.dat.find_aliases(name=name)

    @restmethod("GET", "/hosts")
    async def read_hosts_all(cls):
        """
        Returns all available resource hosts.
        """

        return cls.dat.find_hosts()

    @restmethod("GET", "/hosts/{name}")
    async def read_hosts_one(cls, name: str):
        """
        Returns all hosts that match a name.
        """

        return cls.dat.find_hosts(name=name)

    @restmethod("GET", "/resources/")
    async def read_resource_all(cls):
        """Returns all available resources."""

        return cls.dat.find_resources()

    @restmethod("PUT", "/host/{name}", status_code=201)
    async def push_hosts_one(cls, name: str, host: str):
        """
        Creates a new, or updates an existing,
        host.
        """

        with handle_db_action():
            cls.dat.push_hosts_multi(dict(name=name, host=host))
