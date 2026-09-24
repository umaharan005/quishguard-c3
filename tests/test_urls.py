from quishguard.data.prepare import split_of
from quishguard.data.urls import get_host, normalize_url, registered_domain


def test_normalize_removes_scheme_and_www():
    assert normalize_url("HTTPS://WWW.Example.com/Login?A=B") == "example.com/Login?A=B"
    assert normalize_url("hhttps://evil.xyz/") == "evil.xyz"
    assert normalize_url("example.com/") == "example.com"


def test_scheme_does_not_change_normalised_form():
    assert normalize_url("http://paypal-login.top/x") == normalize_url("paypal-login.top/x")


def test_host_and_domain():
    assert get_host("http://login.paypal.com.evil.co.uk:8080/a") == "login.paypal.com.evil.co.uk"
    assert registered_domain("login.paypal.com.evil.co.uk") == "evil.co.uk"
    assert registered_domain("87.120.115.240") == "87.120.115.240"
    assert get_host("www.sliit.lk/courses") == "sliit.lk"


def test_bad_ipv6_does_not_crash():
    assert isinstance(get_host("http://[bad/url"), str)


def test_split_is_stable_per_domain():
    assert split_of("example.com") == split_of("example.com")
    assert split_of("example.com") in {"train", "val", "test"}


def test_mask_tld():
    from quishguard.data.urls import mask_tld
    assert mask_tld("shop.example.lk/menu") == "shop.example.<tld>/menu"
    assert mask_tld("87.1.2.3:80/i") == "87.1.2.3:80/i"
