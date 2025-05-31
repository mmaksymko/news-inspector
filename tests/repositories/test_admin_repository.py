import pytest
from types import SimpleNamespace
from datetime import datetime, timedelta

from repositories.admin_repository import (
    is_admin,
    add_admin,
    Admin,
)

import repositories.admin_repository as admin_mod


class FakeQueryForIsAdmin:
    """
    A fake “query” object that emulates:
        session.query(Admin).filter_by(handle=...).scalar()
    We can preconfigure it to return either a non-None value (to simulate “found”)
    or None (to simulate “not found”).
    """
    def __init__(self, return_for_scalar):
        # return_for_scalar is either some non‐None object or None
        self._return = return_for_scalar
        self._filter_handle = None

    def filter_by(self, handle):
        # record the handle if you want, then return self
        self._filter_handle = handle
        return self

    def scalar(self):
        # Return whatever was preconfigured
        return self._return


class FakeSessionForIsAdmin:
    """
    A fake Session that, when .query(Admin) is called, returns a FakeQueryForIsAdmin.
    """
    def __init__(self, return_for_scalar):
        self.return_for_scalar = return_for_scalar
        self.queried_with = []  # record what model was passed into query()

    def query(self, model):
        self.queried_with.append(model)
        return FakeQueryForIsAdmin(self.return_for_scalar)


def test_is_admin_returns_true_when_scalar_non_none(monkeypatch):
    """
    If session.query(Admin).filter_by(handle=...).scalar() returns a non‐None value,
    then is_admin(...) should return True.
    """
    # 1) Create a fake session where scalar() returns a dummy object (e.g. Admin instance)
    fake_admin_obj = SimpleNamespace(handle="alice", id=1)
    fake_session = FakeSessionForIsAdmin(return_for_scalar=fake_admin_obj)

    # 2) Bypass @with_session by calling __wrapped__ directly
    result = is_admin.__wrapped__(handle="alice", session=fake_session)

    # 3) Assert that result is True, and that query was indeed called with Admin
    assert result is True
    assert fake_session.queried_with == [Admin]


def test_is_admin_returns_false_when_scalar_none(monkeypatch):
    """
    If session.query(Admin).filter_by(handle=...).scalar() returns None,
    then is_admin(...) should return False.
    """
    fake_session = FakeSessionForIsAdmin(return_for_scalar=None)
    result = is_admin.__wrapped__(handle="bob", session=fake_session)
    assert result is False
    assert fake_session.queried_with == [Admin]


class FakeSessionForAddAdmin:
    """
    A fake Session that records calls to .add(obj). We don’t need to emulate flush() or commit()
    because add_admin only does session.add(admin) and returns it.
    """
    def __init__(self):
        self.added = []  # will collect any objects passed to .add()

    def add(self, obj):
        self.added.append(obj)


def test_add_admin_creates_and_adds_admin(monkeypatch):
    """
    When add_admin(handle) is called:
      • It should construct an Admin instance whose .handle == the provided handle.
      • It should call session.add(...) exactly once with that Admin instance.
      • It should return that new Admin object.
    """
    fake_session = FakeSessionForAddAdmin()

    # Call the function, bypassing @with_session
    new_admin = add_admin.__wrapped__(handle="charlie", session=fake_session)

    # Verify that `new_admin` is an Admin (or at least has the right attribute)
    assert isinstance(new_admin, Admin)
    assert new_admin.handle == "charlie"

    # Verify that session.add was called exactly once with new_admin
    assert fake_session.added == [new_admin]


class FakeResultProxy:
    """
    A fake object returned by session.execute(stmt) whose .all() method yields
    exactly the list we configured. For example, [("cat1", 42), ...].
    """
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeSessionForGetStats:
    """
    A fake Session that captures the Select statement passed into execute(...)
    and returns a FakeResultProxy(rows).
    """
    def __init__(self, return_rows):
        # return_rows is the list that .execute(stmt).all() should produce.
        self.return_rows = return_rows
        self.executed_statements = []  # will collect each stmt passed to .execute()

    def execute(self, stmt):
        # Record the statement
        self.executed_statements.append(stmt)
        # Return a fake result proxy whose .all() returns return_rows
        return FakeResultProxy(self.return_rows)


def test_get_stats_returns_rows_from_execute(monkeypatch):
    """
    If session.execute(...) yields a proxy whose .all() returns a list of tuples,
    then get_stats(...) should simply return that same list.

    We also check that the 'since' filter is honored (i.e. the passed datetime).
    """
    # 1) Prepare dummy data that we want get_stats to return
    dummy_rows = [
        ("CategoryA", 10),
        ("CategoryB", 5),
        ("CategoryC", 2),
    ]
    fake_session = FakeSessionForGetStats(return_rows=dummy_rows)

    # 2) Choose an expl
