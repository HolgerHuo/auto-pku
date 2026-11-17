import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from browser_session import BrowserSession


def test_browser_owner_and_private_export(tmp_path, monkeypatch):
    b = BrowserSession("http://127.0.0.1:9377", "secret", "persistent-owner")
    calls = []

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"tabs": []}

    def request(method, url, **kwargs):
        calls.append(kwargs)
        return Reply()

    monkeypatch.setattr(b.client, "request", request)
    assert b.tabs() == []
    assert calls[0]["params"] == {"userId": "persistent-owner"}
    output = tmp_path / "cookies.txt"
    b.export(
        [
            {"domain": ".pku.edu.cn", "name": "session", "value": "secret"},
            {"domain": ".example.com", "name": "other", "value": "omit"},
        ],
        output,
    )
    assert output.stat().st_mode & 0o777 == 0o600
    assert "omit" not in output.read_text()


def test_recovery_preserves_session_cookies_and_other_tabs(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import browser_session

    b = BrowserSession("http://127.0.0.1:9377", "secret", "owner")
    tabs = [{"tabId": "original", "listItemId": "user"}]
    imported = []
    deleted = []
    group = None

    def call(method, path, data=None):
        nonlocal group
        if path == "/tabs" and method == "GET":
            return {"tabs": tabs.copy()}
        if path == "/tabs" and method == "POST":
            group = data["sessionKey"]
            tabs.append({"tabId": "home", "listItemId": group})
            return {"tabId": "home"}
        if path.endswith("/evaluate"):
            return {"result": True}
        if path.endswith("/click"):
            if "playVideo" in data["selector"]:
                tabs.append(
                    {
                        "tabId": "play",
                        "listItemId": group,
                        "url": "https://course.pku.edu.cn/playVideo.action?token=fixture",
                    }
                )
            return {"ok": True}
        if path.endswith("/storage_state"):
            return {
                "cookies": [
                    {
                        "name": "JSESSIONID",
                        "value": "session",
                        "domain": "course.pku.edu.cn",
                        "path": "/",
                        "expires": -1,
                    },
                    {
                        "name": "_token",
                        "value": "token",
                        "domain": ".pku.edu.cn",
                        "path": "/",
                        "expires": -1,
                    },
                ]
            }
        if path.endswith("/cookies"):
            imported.extend(data["cookies"])
            tabs.append({"tabId": "concurrent-user-tab", "listItemId": "user"})
            return {"ok": True}
        if method == "DELETE":
            deleted.append(path.split("/")[-1])
            return {"ok": True}
        raise AssertionError((method, path))

    source = (
        "https://yjapise.pku.edu.cn/casapi/index.php?r=auth/login-with-sign"
        "&forward=https://onlineroomse.pku.edu.cn/player?course_id%3D10%26sub_id%3D1&timestamp=123"
    )

    def make_session(jar, proxy):
        def get(url, **kwargs):
            if "login-with-sign" in url:
                assert "&timestamp=123" in url
            return SimpleNamespace(
                status_code=200,
                text=f'<iframe src="{source}"></iframe>',
                raise_for_status=lambda: None,
                json=lambda: {"code": 0},
            )

        return SimpleNamespace(cookies=jar, get=get)

    monkeypatch.setattr(b, "call", call)
    monkeypatch.setattr(browser_session, "session", make_session)
    b.recover("_10_1", tmp_path / "cookies.txt")
    assert all(c["expires"] == -1 for c in imported)
    assert set(deleted) == {"home", "play"}
    assert b.course_id == 10
