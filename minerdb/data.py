"""
Defined here are ORM Miner models using
SQLAlchemy.
"""

import contextlib, enum, typing

from sqlalchemy import (
    create_engine,
    delete,
    insert,
    select,
    update,
    func,
    Delete,
    Engine,
    Insert,
    Select,
    Update)
from sqlalchemy import (
    Boolean,
    Enum,
    ForeignKey,
    Integer,
    Sequence,
    String,
    Table,
    TypeDecorator)
from sqlalchemy.orm import (
    declared_attr,
    mapped_column,
    relationship,
    DeclarativeBase,
    Mapped,
    Session)

from minerdb.error import RecordExists, RecordNotFound

Tt         = typing.TypeVar("Tt")
Ps         = typing.ParamSpec("Ps")
MDt        = typing.TypeVar("MDt", bound="MinerDB")
MMt        = typing.TypeVar("MMt", bound="MinerModelV1")
StmtT      = typing.TypeVar("StmtT", Select, Update, Insert, Delete)
DRCallable = typing.Callable[Ps, "DataResponse[MMt]"]

OperationMap = {
    "==": "__eq__",
    ">=": "__ge__",
    "<=": "__le__",
    ">":  "__gt__",
    "<":  "__lt__"
}

def data_drop(count: typing.Optional[int] = None) -> "DataResponse":
    """Create a data response for DROP call."""

    dr = data_new(count, status=DataStatus.DROPPED)
    dr.pop("result")
    return dr

def data_find(
    count: typing.Optional[int] = None,
    result: typing.Optional[typing.Iterable] = None) -> "DataResponse":
    """Create a data response for a FIND call."""

    status = DataStatus.FOUND if count else DataStatus.FOUND_NONE
    dr     = data_new(count, result, status)
    return dr


def data_push(count: typing.Optional[int] = None) -> "DataResponse":
    """Create a data response for a PUSH call."""

    status = DataStatus.UPDATED if count else DataStatus.CREATED
    dr     = data_new(count, status=status)
    dr.pop("result")
    return dr


def data_new(
    count: typing.Optional[int] = None,
    result: typing.Optional[typing.Iterable] = None,
    status: typing.Optional["DataStatus"] = None) -> "DataResponse":
    """Create a data response."""

    count  = count or 0
    result = result or iter(())
    status = status or DataStatus.OK

    return DataResponse(count=count, result=result, status=status)


def resource_repr(
    resource: "MinerModelV1",
    fields: dict[str, typing.Any]) -> str:
    """
    Create a string representation of a database
    record.
    """

    name   = resource.__class__.__qualname__
    fnames = ", ".join((f"{k}={v!r}" for k,v in fields.items()))
    return f"{name}(" + fnames + ")"


def version_compare(
    stmt: StmtT,
    model: "HasVersionOrders",
    version: str) -> StmtT:
    """
    Returms a modified version of the original
    statement to include a version comparison.
    """

    op, vers = "__eq__", version
    if vers[:2] in ("==", ">=", "<="):
        op, vers = OperationMap[vers[:2]], vers[2:]
    elif version[:1] in (">", "<"):
        op, vers = OperationMap[vers[:1]], vers[1:]

    parts  = version_parse_str(vers)
    orders = (model.major, model.minor, model.patch)
    for order, part in zip(orders, parts):
        cmp = order.__ge__
        if part >= 0:
            cmp = getattr(order, op)
        stmt = stmt.where(cmp(part))

    return stmt


def validate_pre_update(
    model: MMt,
    func: DRCallable[MMt], #type: ignore
    **kwds) -> "DataResponse[MMt]":
    """
    Check results to ensure an update call will
    not interfere with data integrity.
    """

    found  = func(**kwds)
    count  = found["count"]

    found["result"] = tuple(found["result"])
    if count > 1:
        raise RecordExists("Found too many records.")
    if count > 0 and found["result"][0] == model:
        raise RecordExists("Record exists.")

    return found


def version_parse_str(version: str) -> tuple[int, int, int]:
    """
    Parse a a version string into a 3 integer
    tuple.
    """

    parts = list(map(int, version.split(".", maxsplit=3)))
    ret   = [-1, -1, -1]
    for idx, part in enumerate(parts):
        ret[idx] = part
    return tuple(ret)


class ServiceAPI(enum.StrEnum):
    """Type of service API."""

    Bukkit   = enum.auto()
    Paper    = enum.auto()
    Spigot   = enum.auto()
    Velocity = enum.auto()


class ResourceKind(enum.StrEnum):
    """Type of purpose the resource serves."""

    Plugin    = enum.auto()
    PluginAPI = enum.auto()
    Proxy     = enum.auto()
    Server    = enum.auto()


class DataResponse(typing.TypedDict, typing.Generic[Tt]):
    """Data response from database lookup."""
 
    count:  int
    result: typing.Iterator[Tt]
    status: "DataStatus"


class DataStatus(enum.StrEnum):
    OK         = enum.auto()
    FOUND      = enum.auto()
    FOUND_NONE = enum.auto()
    CREATED    = enum.auto()
    DROPPED    = enum.auto()
    UPDATED    = enum.auto()


class MinerModelV1(DeclarativeBase):
    """
    Some object used in the search and
    procurement of Minecraft server resources.
    """

    __abstract__ = True

    @declared_attr
    def __tablename__(self):
        return "_".join(["mdb", self.__tablename__])


class HasVersionOrders(MinerModelV1):
    """
    Model abstract that has version information.
    """

    __abstract__ = True

    major: Mapped[int] = mapped_column(Integer(), default=0)
    minor: Mapped[int] = mapped_column(Integer(), default=0)
    patch: Mapped[int] = mapped_column(Integer(), default=0)


class Alias(MinerModelV1):
    """
    Name used in place of retrieveable from host.
    """

    __tablename__ = "aliases"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True)
    resource: Mapped[int] = mapped_column(
        ForeignKey("resources.id"),
        primary_key=True)
    name: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Relationships
    resources: Mapped["Resource"] = relationship(
        back_populates="aliases")

    def __repr__(self):
        return resource_repr(
            self,
            dict(name=self.name, resource=self.resource))


class Host(MinerModelV1):
    """Available hosts for aquiring resources."""

    __tablename__ = "hosts"

    id:   Mapped[int] = mapped_column(Integer(), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    host: Mapped[str] = mapped_column(String(128))

    # Relationships
    resource_hosts: Mapped[list["ResourceByHost"]] = relationship(
        back_populates="hosts",
        cascade="all, delete-orphan")

    def __eq__(self, other: typing.Self):
        return self.name == other.name and self.host == other.host

    def __repr__(self):
        return resource_repr(
            self,
            dict(name=self.name, host=self.host))


class MinecraftVersion(HasVersionOrders):
    """
    Available Minecraft version information.
    """

    __tablename__ = "mc_versions"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True)

    # Relationships
    versions: Mapped[list["Version"]] = relationship(
        back_populates="mc_versions"
    )


class Resource(MinerModelV1):
    """
    Data Master for constructing a single
    resource.
    """

    __tablename__ = "resources"

    id:   Mapped[int] = mapped_column(Integer(), primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    type: Mapped[ServiceAPI] = mapped_column(
        Enum(ServiceAPI),
        default=ServiceAPI.Paper)
    kind: Mapped[ResourceKind] = mapped_column(
        Enum(ResourceKind),
        default=ResourceKind.Plugin)

    # Relationships
    aliases: Mapped[list["Alias"]] = relationship(
        back_populates="resources",
        cascade="all, delete-orphan")
    resource_hosts: Mapped[list["ResourceByHost"]] = relationship(
        back_populates="resources",
        cascade="all, delete-orphan")
    versions: Mapped[list["Version"]] = relationship(
        back_populates="resources",
        cascade="all, delete-orphan")

    def __eq__(self, other: typing.Self):
        return all([
            self.id == other.id,
            self.name == other.name,
            self.type == other.type,
            self.kind == other.kind
        ])

    def __repr__(self):
        return resource_repr(self, dict(
            id=self.id,
            name=self.name,
            type=self.type,
            kind=self.kind))


class ResourceByHost(MinerModelV1):
    """Map resources to hosts."""

    __tablename__ = "resource_by_host"

    id: Mapped[int] = mapped_column(Integer(), primary_key=True)
    resource: Mapped[int] = mapped_column(
        ForeignKey("resources.id"),
        primary_key=True)
    host: Mapped[int] = mapped_column(
        ForeignKey("hosts.id"),
        primary_key=True)

    # Relationships
    hosts: Mapped["Host"] = relationship(
        back_populates="resource_hosts")
    resources: Mapped["Resource"] = relationship(
        back_populates="resource_hosts")

    def __repr__(self):
        return resource_repr(
            self,
            dict(resource=self.resource, host=self.host))


class Version(HasVersionOrders):
    """Version of some available resource."""

    __tablename__ = "versions"

    id:          Mapped[int]  = mapped_column(Integer(), primary_key=True)
    resource:    Mapped[int]  = mapped_column(ForeignKey("resources.id"), primary_key=True)
    build:       Mapped[str]  = mapped_column(String(16), nullable=True)
    is_snapshot: Mapped[bool] = mapped_column(Boolean(), default=False)
    mc_version:  Mapped[int]  = mapped_column(ForeignKey("mc_versions.id"))

    # Relationships
    resources: Mapped["Resource"] = relationship(
        back_populates="versions")
    mc_versions: Mapped[MinecraftVersion] = relationship(
        back_populates="versions")

    def __repr__(self):
        return resource_repr(
            self,
            dict(id=self.id, resource=self.resource))


class MinerDB(typing.Protocol):
    """MinerDB Interface Object."""

    @property
    def engine(self) -> Engine:
        """Underlying SQLAlchemy engine."""

    @property
    def root_model(self) -> type[MinerModelV1]:
        """MinerDB base object."""

    def create(self, model: MinerModelV1) -> Insert[MinerModelV1]:
        """Initialize an insert statement."""

    def count(self, statement: Select[tuple[MinerModelV1]]) -> int:
        """
        Get the result count from a `Select`
        statement.
        """

    def delete(self, model: MinerModelV1) -> Delete[MinerModelV1]:
        """Initializd a delete statement."""

    def push(self, *inst: MinerModelV1, commit: bool = ...):
        """
        Performs some action against the database.
        """

    def find(
        self,
        statement: Select[tuple[MinerModelV1]]) -> typing.Iterator[MinerModelV1]:
        """
        Executes a statement returning the result.
        """

    def init(self) -> None:
        """Initialize MirrirDB."""

    def select(
        self,
        *model: type[MinerModelV1]) -> Select[tuple[MinerModelV1]]:
        """Initialize a select statement."""

    def session(
        self,
        *,
        commit: bool = ...) -> typing.ContextManager[Session]:
        """Database session."""

    def update(self, model: Tt) -> Update[Tt]:
        """Initialize an update statement."""

    def __init__(self, *, debug: bool = ...) -> None:
        """Create an instance of `MinerDB`."""


class MinerDBSimple(MinerDB):
    """
    Manages an In-Memory instance of MirrirDB.
    """

    @property
    def engine(self):
        return self._engine

    @property
    def root_model(self):
        return self._root_model

    def create(self, model):
        return insert(model)

    def count(self, statement):
        # We have to filter out the joins in
        # the passed statement in order for
        # our count query to work.
        # Otherwise, SQLAlchemy renders the
        # JOINs multiple times leaving column
        # names ambiguous to the DB driver.
        froms = [f for f in statement.froms if isinstance(f, Table)]
        count = (statement
            .with_only_columns(func.count())
            .select_from(*froms)
            .order_by(None))

        with self.session() as sxn:
            return sxn.execute(count).scalar()

    def delete(self, model):
        return delete(model)

    def push(self, statement, commit=False):
        with self.session(commit=commit) as sxn:
            sxn.execute(statement)

    def find(self, statement):
        with self.session() as sxn:
            for row in sxn.scalars(statement):
                yield row

    def init(self):
        self.root_model.metadata.create_all(self.engine)

    def select(self, *model):
        return select(*model)

    @contextlib.contextmanager
    def session(self, *, commit=False):
        with Session(self.engine) as sxn:
            yield sxn

            if commit:
                sxn.commit()

    def update(self, model):
        return update(model)

    def __init__(self, *, debug=False):
        self._name       = "miner.sqlite"
        self._engine     = create_engine(f"sqlite:///{self._name}", echo=debug)
        self._root_model = MinerModelV1


class MinerDB_V1(MinerDBSimple):
    """MinerDB version: 1"""

    def drop_alias(self, id: int) -> DataResponse:
        """
        Attempt to delete an instance of `Alias`.
        """

        self.push(
            self.delete(Alias).where(Alias.id == id),
            commit=True)
        return data_drop(1)

    def drop_host(self, id: int) -> DataResponse:
        """
        Attempt to delete an instance of `Host`.
        """

        self.push(
            self.delete(Host).where(Host.id == id),
            commit=True)
        return data_drop(1)

    def drop_minecraft(self, id: int) -> DataResponse:
        """
        Attempt to delete an instance of
        `MinecraftVersion`.
        """

        self.push(
            self.delete(MinecraftVersion).where(MinecraftVersion.id == id),
            commit=True)
        return data_drop(1)

    def drop_resource(self, id: int) -> DataResponse:
        """
        Attempt to delete an instance of
        `Resource`.
        """

        self.push(
            self.delete(Resource).where(Resource.id == id),
            commit=True)
        return data_drop(1)

    def drop_version(self, id: int) -> DataResponse:
        """
        Attempt to delete an instance of
        `Version`.
        """

        self.push(
            self.delete(Version).where(Version.id == id),
            commit=True)

    def find_aliases(
        self,
        id: int | None = None,
        name: str | None = None,
        resource: int | None = None) -> DataResponse[Alias]:
        """
        Query database for instances of `Alias`.
        """

        stmt = self.select(Alias)
        if id is not None:
            stmt = stmt.where(Alias.id == id)
        if resource is not None:
            stmt = (stmt
                .join(Resource, Alias.resource == Resource.id)
                .where(Alias.resource == resource))
        if name is not None:
            stmt = stmt.where(Alias.name.contains(name))

        return data_find(self.count(stmt), self.find(stmt))

    def find_hosts(
        self,
        id: int | None = None,
        name: str | None = None,
        resource: int | None = None) -> DataResponse[Host]:
        """
        Query database for instances of `Host`.
        """

        stmt = self.select(Host)
        if id is not None:
            stmt = stmt.where(Host.id == id)
        if resource is not None:
            stmt = (stmt
                .join(ResourceByHost, Host.id == ResourceByHost.host)
                .where(ResourceByHost.resource == resource))
        if name is not None:
            stmt = stmt.where(Host.name.contains(name))

        return data_find(self.count(stmt), self.find(stmt))

    def find_minecraft(
        self,
        id: int | None = None,
        version: typing.Optional[str] = None):
        """
        Query database for instances of
        `MinecraftVersion`.
        """

        stmt = self.select(MinecraftVersion)
        if id is not None:
            stmt = stmt.where(MinecraftVersion.id == id)
        if version is not None:
            stmt = version_compare(stmt, MinecraftVersion, version)

        return data_find(self.count(stmt), self.find(stmt))

    def find_resources(
        self,
        id: int | None = None,
        name: str | None = None,
        type: ServiceAPI | None = None,
        kind: ResourceKind | None = None) -> DataResponse[Resource]:
        """
        Query database for instances of
        `Resource`.
        """

        stmt = self.select(Resource)
        if id is not None:
            stmt = stmt.where(Resource.id == id)
        if name is not None:
            stmt = stmt.where(Resource.name.contains(name))
        if type is not None:
            stmt = stmt.where(Resource.type == type)
        if kind is not None:
            stmt = stmt.where(Resource.kind == kind)

        return data_find(self.count(stmt), self.find(stmt))

    def find_versions(
        self,
        id: int | None = None,
        resource: int | None = None,
        version: str | None = None,
        build: str | None = None,
        compatibility: str | None = None,
        is_snapshot: bool | None = None) -> DataResponse[Version]:
        """
        Query database for instances of `Version`.
        """

        stmt = self.select(Version)
        if id is not None:
            stmt = stmt.where(Version.id == id)
        if resource is not None:
            stmt = (stmt
                .join(Resource, Resource.id == Version.resource)
                .where(Resource.id == resource))
        if version is not None:
            stmt = version_compare(stmt, Version, version)
        if build is not None:
            stmt = stmt.where(Version.build == build)
        if compatibility is not None:
            # Might not work without using join
            # clause.
            stmt = version_compare(stmt, MinecraftVersion, compatibility)
        if is_snapshot is not None:
            stmt = stmt.where(Version.is_snapshot == is_snapshot)

        return data_find(self.count(stmt), self.find(stmt))

    def make_alias(self, meta: dict[str, str]) -> DataResponse:
        """Create a new alias entry."""

        model, count = self.prevalidate_alias(meta)
        if count == 1:
            raise RecordExists("Alias exists.")

        self.push(
            self.create(Alias)
                .values(
                    resource=model.resource,
                    name=model.name),
            commit=True)

        return data_push(count)

    def make_host(self, meta: dict[str, str]) -> DataResponse:
        """Create a new host entry."""

        model, count = self.prevalidate_host(meta)
        if count == 1:
            raise RecordExists("Host exists.")

        self.push(
            self.create(Host)
                .values(host=model.host, name=model.name),
            commit=True)
    
        return data_push(count)

    def make_minecraft(self, meta: dict[str, str]) -> DataResponse:
        """Create a new Minecraft entry."""

        model, count = self.prevalidate_minecraft(meta)
        if count == 1:
            raise RecordExists("Minecraft exists.")

        self.push(
            self.create(MinecraftVersion)
                .values(
                    major=model.major,
                    minor=model.minor,
                    patch=model.patch),
            commit=True)

        return data_push(count)

    def make_resource(self, meta: dict[str, str]) -> DataResponse:
        """Create a new resource entry."""

        model, count = self.prevalidate_resource(meta)
        if count == 1:
            raise RecordExists("Resource exists.")

        self.push(
            self.create(Resource)
                .values(
                    name=model.name,
                    type=model.type,
                    kind=model.kind),
                commit=True)

        return data_push(count)

    def prevalidate_alias(self, meta: dict[str, str]) -> tuple[Alias, int]:
        """
        Validate a alias entry before it can be
        modified or created. Raises a
        `MinerException` if invalid, returns a
        query model and the result count
        otherwise.
        """

        owner_count = self.find_resources(meta["resource"])["count"]
        if owner_count > 1:
            raise RecordExists("Found too many resources.")
        if owner_count < 1:
            raise RecordNotFound("No resource exists.")

        model = Alias(**meta)
        found = validate_pre_update(
            model,
            self.find_aliases,
            resource=meta["resource"],
            name=meta.get("name", None))
        count = found["count"]

        if count > 0:
            model.id = found["result"][0].id
        return model, found["count"]

    def prevalidate_host(self, meta: dict[str, str]) -> tuple[Host, int]:
        """
        Validate a host entry before it can be
        modified or created. Raises a
        `MinerException` if invalid, returns a
        query model and the result count
        otherwise.
        """

        model = Host(**meta)
        found = validate_pre_update(
            model,
            self.find_hosts,
            id=meta.get("id", None),
            name=meta.get("name", None),
            resource=meta.get("resource", None))
        count = found["count"]

        if count > 0:
            model.id = found["result"][0].id
        return model, found["count"]

    def prevalidate_minecraft(
        self,
        meta: dict[str, str]) -> tuple[MinecraftVersion, int]:
        """
        Validate a Minecraft entry before it can
        be modified or created. Raises a
        `MinerException` if invalid, returns a
        query model and the result count
        otherwise.
        """ 

        maj, min, pat = version_parse_str(meta["version"])
        model = MinecraftVersion(major=maj, minor=min, patch=pat)
        found = validate_pre_update(
            model,
            self.find_minecraft,
            id=meta.get("id", None),
            version=meta.get("version", None))
        count = found["count"]

        if count > 0:
            model.id = found["result"][0].id
        return model, found["count"]

    def prevalidate_resource(
        self,
        meta: dict[str, str]) -> tuple[Resource, int]:
        """
        Validate a resource entry before it can
        be modified or created. Raises a
        `MinerException` if invalid, returns a
        query model and the result count
        otherwise.
        """

        model = Resource(**meta)
        found = validate_pre_update(
            model,
            self.find_resources,
            id=meta.get("id", None),
            name=meta.get("name", None))
        count = found["count"]

        if count > 0:
            model.id = found["result"][0].id
        return model, found["count"]

    def push_alias(self, meta: dict[str, str]) -> DataResponse:
        """Update an existing alias entry."""

        model, count = self.prevalidate_alias(meta)
        if count < 1:
            raise RecordNotFound("Alias not found.")

        self.push(
            self.update(Alias).values(name=model.name).where(id=model.id),
            commit=True)

        return data_push(count)

    def push_host(self, meta: dict[str, str]) -> DataResponse:
        """Update an existing host entry."""

        model, count = self.prevalidate_host(meta)
        if count < 1:
            raise RecordNotFound("Host not found.")

        self.push(
            self.update(Host)
                .values(host=model.host)
                .where(Host.id == model.id),
            commit=True)
        
        return data_push(count)

    def push_resource(self, meta: dict[str, str]) -> DataResponse:
        """
        Update an existing resource.
        """

        model, count = self.prevalidate_resource(meta)

        self.push(
            self.update(Resource)
                .values(
                    type=model.type,
                    kind=model.kind)
                .where(Resource.id == model.id),
            commit=True)

        return data_push(count)


if __name__ == "__main__":
    minerdb = MinerDB_V1(debug=True)
    minerdb.init()
    cnt, aliases = minerdb.find_aliases()
