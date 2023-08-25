"""
Defined here are ORM Miner models using
SQLAlchemy.
"""

import contextlib, enum, pathlib, typing

from sqlalchemy import (
    create_engine, insert, select, update, func, Engine, Insert, Select, Update)
from sqlalchemy import Boolean, Enum, ForeignKey, Integer, String
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from error import RecordExists, RecordNotFound

T = typing.TypeVar("T")


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


class Service(enum.StrEnum):
    """Type of service."""

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


class DataResponse(typing.TypedDict, typing.Generic[T]):
    """Data response from database lookup."""
 
    count:  int
    result: typing.Iterator[T]


class MinerModelV1(DeclarativeBase):
    """
    Some object used in the search and
    procurement of Minecraft server resources.
    """


class Alias(MinerModelV1):
    """
    Name used in place of retrieveable from host.
    """

    __tablename__ = "mdb_aliases"

    resource: Mapped[int] = mapped_column(
        ForeignKey("mdb_resources.id"),
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

    __tablename__ = "mdb_hosts"

    name: Mapped[str] = mapped_column(primary_key=True)
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


class Resource(MinerModelV1):
    """
    Data Master for constructing a single
    resource.
    """

    __tablename__ = "mdb_resources"

    id:   Mapped[int]     = mapped_column(primary_key=True)
    name: Mapped[str]     = mapped_column(String(64))
    type: Mapped[Service] = mapped_column(
        Enum(Service),
        default=Service.Paper)
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

    __tablename__ = "mdb_resource_by_host"

    resource: Mapped[Resource] = mapped_column(
        ForeignKey("mdb_resources.id"),
        primary_key=True)
    host: Mapped[Host] = mapped_column(
        ForeignKey("mdb_hosts.name"),
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


class Version(MinerModelV1):
    """Version of some available resource."""

    __tablename__ = "mdb_versions"

    id:          Mapped[int]  = mapped_column(primary_key=True)
    resource:    Mapped[int]  = mapped_column(ForeignKey("mdb_resources.id"))
    major:       Mapped[int]  = mapped_column(Integer(), default=0)
    minor:       Mapped[int]  = mapped_column(Integer(), default=0)
    patch:       Mapped[str]  = mapped_column(String(16), nullable=True)
    build:       Mapped[str]  = mapped_column(String(16), nullable=True)
    is_snapshot: Mapped[bool] = mapped_column(Boolean(), default=False)

    # Relationships
    resources: Mapped["Resource"] = relationship(back_populates="versions")

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

    def push(self, *inst: MinerModelV1, commit: bool = ...):
        """
        Performs some action against the database.
        """

    def find(
        self,
        statement: Select[tuple[MinerModelV1]]) -> typing.Iterator[MinerModelV1]:
        """Executes a statement."""

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

    def update(self, model: T) -> Update[T]:
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
        with self.session() as sxn:
            counter = (statement
                .with_only_columns(func.count())
                .select_from(*statement.froms)
                .order_by(None))
            return sxn.execute(counter).scalar()

    def push(self, statement, commit=False):
        with self.session(commit=commit) as sxn:
            sxn.execute(statement)

    def find(self, statement):
        with self.session() as sxn:
            for row in sxn.scalars(statement):
                yield row

    def init(self):
        if (pathlib.Path.cwd() / self._name).exists():
            return
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
        self._name       = "miner.db"
        self._engine     = create_engine(f"sqlite:///{self._name}", echo=debug)
        self._root_model = MinerModelV1


class MinerDB_V1(MinerDBSimple):
    """MinerDB version: 1"""

    def find_aliases(
        self,
        name: str | None = None,
        resource: int | None = None) -> DataResponse[Alias]:
        """
        Query database for instances of `Alias`.
        """

        stmt = self.select(Alias)
        if resource:
            stmt = stmt.join(Resource, Resource.id == resource)
        if name:
            stmt = stmt.where(Alias.name.contains(name))

        return {"count": self.count(stmt), "result": self.find(stmt)}

    def find_hosts(
        self,
        name: str | None = None,
        resource: int | None = None) -> DataResponse[Host]:
        """
        Query database for instances of `Host`.
        """

        stmt = self.select(Host)
        if resource:
            stmt = stmt.join(ResourceByHost, Resource.id == resource)
        if name:
            stmt = stmt.where(Host.name.contains(name))

        return {"count": self.count(stmt), "result": self.find(stmt)}

    def find_resources(
        self,
        resource: int | None = None) -> DataResponse[Resource]:
        """
        Query database for instances of
        `Resource`.
        """

        stmt = self.select(Resource)

        return {"count": self.count(stmt), "result": self.find(stmt)}

    def push_hosts_multi(self, *host: dict[str, str] | Host):
        """
        Creates new, or updates existing,
        hosts.
        """

        host = tuple(Host(**h) if isinstance(h, dict) else h for h in host)
        for h in host:
            self.push_hosts_once(h)

    def push_hosts_once(self, host: dict[str, str] | Host):
        """
        Creates a new, or updates an existing,
        host.
        """

        host   = Host(**host) if isinstance(host, dict) else host
        found  = self.find_hosts(host.name)
        count  = found["count"]
        result = found["result"]

        if count > 1:
            raise RecordExists(f"found too many records.")
        if count > 0 and next(result) == host:
            raise RecordExists(f"name exists.")
        if count == 1:
            stmt = (self.update(Host)
                .values(host=host.host)
                .where(Host.name == host.name))
        else:
            stmt = (self.create(Host)
                .values(host=host.host, name=host.name))
        self.push(stmt, commit=True)


if __name__ == "__main__":
    minerdb = MinerDB_V1(debug=True)
    minerdb.init()
    cnt, aliases = minerdb.aliases()
