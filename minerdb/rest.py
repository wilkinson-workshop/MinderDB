"""Defines Web APIs."""

import abc, contextlib, functools, inspect, typing

from fastapi import FastAPI, APIRouter
from fastapi.exceptions import HTTPException
from fastapi.types import DecoratedCallable

from minerdb.data import MDt, ServiceAPI, MinerDB_V1, ResourceKind
from minerdb.error import MinerException

Tt = typing.TypeVar("Tt")
Rt = typing.TypeVar("Rt", covariant=True)
Ps = typing.ParamSpec("Ps")
DCWrapper = typing.Callable[[DecoratedCallable], DecoratedCallable]
MethodCallable = typing.Callable[typing.Concatenate[Tt, Ps], Rt]


def minerdb_exc_handler(fn: DecoratedCallable) -> DecoratedCallable:
    """
    Opens a context to handle `MinerException`s
    and raise them as `HTTPException`s.
    """

    @contextlib.contextmanager
    def inner_context():
        try:
            yield
        except MinerException as exc:
            raise HTTPException(exc.status_code or 500, exc.message) from exc

    def inner(*args, **kwds):
        with inner_context():
            return fn(*args, **kwds)

    async def ainner(*args, **kwds):
        with inner_context():
            return await fn(*args, **kwds)

    wrapper = inner
    if inspect.iscoroutinefunction(fn):
        wrapper = ainner

    return functools.update_wrapper(wrapper, fn)


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
class RESTMethod(typing.Protocol, MethodCallable[Tt, Ps, Rt]):
    """
    Method protocol for wrapped callables that
    are registered as REST implementations.
    """

    __is_restmethod__:   bool
    __restmethod_meta__: tuple[str, str, typing.Mapping]
    __wrapped__:         MethodCallable[Tt, Ps, Rt]

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

    def wrapped(self, owner: typing.Any = None) -> MethodCallable[Tt, Ps, Rt]:
        """Wrapped REST implementation."""

        if owner:
            return functools.partial(self.__wrapped__, owner)
        return self.__wrapped__

    def __init__(
        self,
        fn: MethodCallable[Tt, Ps, Rt],
        method: str,
        path: str, 
        **kwds):
        """Wraps a callable as a `RestMethod`"""

        kwds["description"] = kwds.get("description", fn.__doc__)
        kwds["methods"] = kwds.get("methods", [method])
        kwds["name"] = kwds.get("name", fn.__name__)

        self.__is_restmethod__   = True
        self.__restmethod_meta__ = (method, path, kwds)
        self.__wrapped__         = fn

    def __call__(self, owner, *args, **kwds) -> Rt:
        return self.wrapped(owner, *args, **kwds)


class MinerAPINamespace(typing.TypedDict):
    """
    Namespace values used in type initialization.
    """

    __data_class__:   type[MDt]
    __rest_class__:   type[FastAPI]
    __data__:         MDt
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


class MinerAPI(typing.Protocol[MDt], metaclass=MinerAPI_Meta):
    """MinerDB API interface object."""

    __data_class__:   type[MDt]
    __rest_class__:   type[FastAPI]
    __data__:         MDt
    __debug_mode__:   bool
    __rest__:         FastAPI
    __route_prefix__: str
    __router__:       APIRouter
    __routes__:       typing.Sequence[str]

    @property
    def data(self) -> MDt:
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
        kwds["title"] = kwds.get("title", "MinerDB")

        self.__data__ = self.__data_class__(debug=self.__debug_mode__)
        self.__rest__ = self.__rest_class__(debug=self.__debug_mode__, **kwds)

        route: RESTMethod
        for name in self.routes:
            route = getattr(self, name)
            args  = route.path, route.wrapped(self)
            self.router.add_api_route(*args, **route.options)

        self.rest.include_router(self.router)
        self.data.init()


class MinerAPI_V1(MinerAPI, prefix="/v1", debug=True):
    """MinerDB API Version: 1"""

    if typing.TYPE_CHECKING:
        data: MinerDB_V1

    @restmethod("GET", "/")
    async def read_root(self):
        """Root of MinerDB API v1."""

        return {"status": "OK", "version": 1}

    @restmethod("GET", "/alias")
    @minerdb_exc_handler
    async def read_aliases(self, name: typing.Optional[str] = None):
        """
        Returns all available resource aliases.
        """

        return self.data.find_aliases(name)

    @restmethod("GET", "/host")
    @minerdb_exc_handler
    async def read_hosts(self, name: typing.Optional[str] = None):
        """
        Returns all hosts that match a name.
        """

        return self.data.find_hosts(name)

    @restmethod("GET", "/minecraft")
    @minerdb_exc_handler
    async def read_minecraft(self, version: typing.Optional[str] = None):
        """Returns all minecraft metadata."""

        return self.data.find_minecraft(version=version)

    @restmethod("GET", "/resource")
    @minerdb_exc_handler
    async def read_resources(
        self,
        id: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        type: typing.Optional[ServiceAPI] = None,
        kind: typing.Optional[ResourceKind] = None):
        """Returns all available resources."""

        return self.data.find_resources(id, name, type, kind)

    @restmethod("GET", "/version")
    @minerdb_exc_handler
    async def read_versions(
        self,
        resource: int | None = None,
        version: str | None = None,
        build: str | None = None,
        compatibility: str | None = None,
        is_snapshot: bool | None = None):
        """Returns all available versions."""

        return self.data.find_versions(
            resource,
            version,
            build,
            compatibility,
            is_snapshot)

    @restmethod("PATCH", "/host", status_code=201)
    @minerdb_exc_handler
    async def push_host(self, name: str, host: typing.Optional[str] = None):
        """
        Creates a new, or updates an existing,
        host.
        """

        return self.data.push_host(dict(name=name, host=host))

    @restmethod("POST", "/minecraft", status_code=201)
    @minerdb_exc_handler
    async def push_minecraft(self, version: str):
        """Creates a new Minecraft metadata."""

        return self.data.make_minecraft(dict(version=version))

    @restmethod("PATCH", "/resource", status_code=201)
    @minerdb_exc_handler
    async def push_resource(
        self,
        name: str,
        type: ServiceAPI, kind: ResourceKind):
        """
        Creates a new, or updates an existing,
        host.
        """

        return self.data.push_resource(dict(name=name, type=type, kind=kind))


minerdb = MinerAPI_V1().rest
