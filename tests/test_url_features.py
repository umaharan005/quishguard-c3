import numpy as np

from quishguard.url_model.features import UrlFeaturizer, extract_one


def test_scheme_does_not_change_features():
    a = extract_one("http://www.paypal-verify.top/login.php?id=1")
    b = extract_one("paypal-verify.top/login.php?id=1")
    assert a == b


def test_ip_port_and_download():
    f = extract_one("http://175.165.115.126:35682/Mozi.m")
    assert f["is_ip"] == 1 and f["has_port"] == 1 and f["risky_ext"] == 1


def test_free_host_and_words():
    f = extract_one("secure-login.duckdns.org/verify/account")
    assert f["is_free_host"] == 1
    assert f["n_suspicious_words"] >= 3


def test_featurizer_tld_risk_learned_from_train_only():
    urls = ["a.top/x", "b.top/y", "c.com/z", "d.com/w"]
    y = np.array([1, 1, 0, 0])
    fz = UrlFeaturizer(smoothing=0).fit(UrlFeaturizer.raw_frame(urls), y)
    X = fz.transform(UrlFeaturizer.raw_frame(["new.top/a", "new.com/b", "new.zzz/c"]))
    assert X["tld_risk"].tolist() == [1.0, 0.0, 0.5]  # unseen TLD -> prior
    assert X.shape[1] == len(fz.feature_names_)
