"""Defines Web APIs."""

import abc, contextlib, functools, typing

from fastapi import FastAPI, APIRouter
from fastapi.exceptions import HTTPException
from fastapi.types import DecoratedCallable

from minerdb.data import MinerDB, MinerDB_V1
from minerdb.error import MinerException

T = typing.TypeVar("T")
R = typing.TypeVar("R", covariant=True)
P = typing.ParamSpec("P")
DCWrapper = typing.Callable[[DecoratedCallable], DecoratedCallable]
MethodCallable = typing.Callable[typing.Concatenate[T, P], R]


@contextlib.contextmanager
def minerdb_exc_handler():
    """
    Opens a context to handle `MinerException`s
    and raise them as `HTTPException`s.
    """

    try:
        yield
    except MinerException as exc:
        raise HTTPException(exc.status_code or 500, exc.message) from exc


def isdunder(name: str) -> bool:
    """
    Whether attribute name is a double underscore
    name.
    """

    return all(
    (
        len(name) > 4,
        name[:2] == name[-2:] == "__",
        name[2] != "_",
        name[-3] != "_"
    ))


def issunder(name: str) -> bool:
    """
    Whether attribute name is a single
    underscore name.
    """

    return all(
    (
        len(name) > 2,
        name[0] == name[-1] == "_",
        name[1:2] != "_",
        name[-2:-1] != "_"
    ))


def ishidden(name: str) -> bool:
    """Whether attribute name is private."""

    return isdunder(name) or issunder(name)


def isrestmethod(fn: DecoratedCallable) -> bool:
    """
    Whether a callable is a registered REST
    method.
    """

    return isinstance(fn, RESTMethod)


def protomethod(fn: DecoratedCallable) -> DecoratedCallable:
    """Marks a method as abstract."""

    return abc.abstractmethod


def restmethod(method: str, path: str, **kwds) -> DCWrapper:
    """
    Indirectly wraps a callable method as a URI
    implementation.
    """

    return functools.partial(RESTMethod, method=method, path=path, **kwds)


@typing.runtime_checkable
class RESTMethod(typing.Protocol, MethodCallable[T, P, R]):
    """
    Method protocol for wrapped callables that
    are registered as REST implementations.
    """

    __is_restmethod__:   bool
    __restmethod_meta__: tuple[str, str, typing.Mapping]
    __wrapped__:         MethodCallable[T, P, R]

    @property
    def wrapped(self) -> MethodCallable[T, P, R]:
        """Wrapped REST implementation."""

        return self.__wrapped__

    @property
    def method(self) -> str:
        """Implemented REST method."""
    
        return self.__restmethod_meta__[0]

    @property
    def options(self) -> typing.Mapping:
        """Construction options."""

        return self.__restmethod_meta__[2]

    @property
    def path(self) -> str:
        """URI path."""
    
        return self.__restmethod_meta__[1]

    def __init__(
        self,
        fn: MethodCallable[T, P, R],
        method: str,
        path: str, 
        **kwds):
        """Wraps a callable as a `RestMethod`"""

        kwds["name"] = kwds.get("name", fn.__name__)
        kwds["methods"] = kwds.get("methods", [method])

        self.__is_restmethod__   = True
        self.__restmethod_meta__ = (method, path, kwds)
        self.__wrapped__         = fn

    def __call__(self, owner, *args, **kwds) -> R:
        return self.wrapped(owner, *args, **kwds)


class MinerAPINamespace(typing.TypedDict):
    """
    Namespace values used in type initialization.
    """

    __data_class__:   type[MinerDB]
    __rest_class__:   type[FastAPI]
    __data__:         MinerDB
    __debug_mode__:   bool
    __rest__:         FastAPI
    __route_prefix__: str
    __router__:       APIRouter
    __routes__:       typing.Sequence[str]


class MinerAPI_Meta(type(typing.Protocol)):
    """
    Manages registered MinerAPI REST methods.
    """

    def __new__(
        mcls,
        name,
        bases,
        namespace,
        *,
        dat_cls=None,
        debug=None,
        app_cls=None,
        prefix=None):

        namespace = MinerAPINamespace(**namespace)
        if "__data_class__" not in namespace or dat_cls:
            namespace["__data_class__"] = dat_cls or MinerDB_V1
        if "__debug_mode__" not in namespace or debug:
            namespace["__debug_mode__"] = debug or False
        if "__rest_class__" not in namespace or app_cls:
            namespace["__rest_class__"] = app_cls or FastAPI
        if "__route_prefix__" not in namespace or prefix:
            namespace["__route_prefix__"] = prefix or ""
        if "__router__" not in namespace:
            namespace["__router__"] = APIRouter(prefix=prefix or "")
        if "__routes__" not in namespace:
            namespace["__routes__"] = ()

        for name, attr in namespace.items():
            if ishidden(name):
                continue
            if isrestmethod(attr):
                namespace["__routes__"] += (name,)

        return super().__new__(mcls, name, bases, namespace)


class MinerAPI(typing.Protocol, metaclass=MinerAPI_Meta):
    """MinerDB API interface object."""

    __data_class__:   type[MinerDB]
    __rest_class__:   type[FastAPI]
    __data__:         MinerDB
    __debug_mode__:   bool
    __rest__:         FastAPI
    __route_prefix__: str
    __router__:       APIRouter
    __routes__:       typing.Sequence[str]

    @property
    def data(self) -> MinerDB:
        """Database driver."""

        return self.__data__

    @property
    def prefix(self) -> str:
        """URI prefix component."""

        return self.__route_prefix__

    @property
    def rest(self) -> FastAPI:
        """REST API driver."""

        return self.__rest__

    @property
    def router(self) -> APIRouter:
        """REST router."""

        return self.__router__

    @property
    def routes(self) -> typing.Sequence[str]:
        """REST routes."""

        return self.__routes__

    def __init__(self, **kwds):
        """Initialize this API implementation."""
        self.__data__ = self.__data_class__(debug=self.__debug_mode__)
        self.__rest__ = self.__rest_class__(debug=self.__debug_mode__, **kwds)

        for name in self.routes:
            route = getattr(self, name)

            func = functools.partial(route.wrapped, self)
            args = route.path, func
            self.router.add_api_route(*args, **route.options)

        self.rest.include_router(self.router)


class MinerAPI_V1(MinerAPI, prefix="/v1", debug=True):
    """MinerDB API Version: 1"""

    @restmethod("GET", "/")
    async def read_root(self):
        """Root of MinerDB API v1."""

        return {"status": "OK", "version": 1}

    @restmethod("GET", "/aliases")
    def read_aliases_all(self):
        """
        Returns all available resource aliases.
        """

        return self.data.find_aliases()
    
    @restmethod("GET", "/aliases/{name}")
    def read_aliases_one(self, name: str):
        """
        Returns all aliases that match a name.
        """

        return self.data.find_aliases(name=name)

    @restmethod("GET", "/hosts")
    async def read_hosts_all(self):
        """
        Returns all available resource hosts.
        """

        return self.data.find_hosts()

    @restmethod("GET", "/hosts/{name}")
    async def read_hosts_one(self, name: str):
        """
        Returns all hosts that match a name.
        """

        return self.data.find_hosts(name=name)

    @restmethod("GET", "/resources/")
    async def read_resource_all(self):
        """Returns all available resources."""

        return self.data.find_resources()

    @restmethod("PUT", "/host/{name}", status_code=201)
    async def push_hosts_one(self, name: str, host: str):
        """
        Creates a new, or updates an existing,
        host.
        """

        with minerdb_exc_handler():
            self.data.push_hosts_multi(dict(name=name, host=host))


minerdb = MinerAPI_V1().rest
