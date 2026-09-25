#!/usr/bin/env python3
"""钉钉推送连通性测试。

用法::

    # 从 config.env 读（推荐）
    .venv/bin/python test_dingtalk.py

    # 或临时用环境变量覆盖
    DINGTALK_WEBHOOK='https://oapi.dingtalk.com/robot/send?access_token=...' \
    DINGTALK_SECRET='SEC...' .venv/bin/python test_dingtalk.py

成功时钉钉返回 errcode=0；常见错误:
    310000  keywords not in content —— 安全设置选了「自定义关键词」但消息里没有该词
    310000  sign not match       —— 密钥不对，或安全设置没开「加签」
    300001  token is not exist   —— Webhook 地址填错
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse

import requests


def load_config(path="config.env"):
    """从 config.env 读取配置（不覆盖已存在的环境变量）。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = val.strip()


def build_signed_url(webhook, secret):
    """按钉钉文档生成加签后的 URL（与 times_astock.py 中逻辑一致）。"""
    if not secret:
        return webhook
    timestamp = str(round(time.time() * 1000))
    string_to_sign = "{}\n{}".format(timestamp, secret)
    hmac_code = hmac.new(
        secret.encode("utf-8"), string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    sign = urllib.parse.quote(base64.b64encode(hmac_code))
    return f"{webhook}&timestamp={timestamp}&sign={sign}"


def send(webhook, secret, text):
    url = build_signed_url(webhook, secret)
    payload = {
        "msgtype": "markdown",
        "markdown": {"title": "AlphaGPT 连通性测试", "text": text},
    }
    resp = requests.post(
        url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=10
    )
    return resp.status_code, resp.json()


def main():
    load_config()
    webhook = os.environ.get("DINGTALK_WEBHOOK", "").strip()
    secret = os.environ.get("DINGTALK_SECRET", "").strip()
    if not webhook:
        print("缺少 DINGTALK_WEBHOOK：请在 config.env 里填写，或用环境变量传入。")
        sys.exit(1)
    print(f"webhook: {webhook[:45]}...{'（含密钥）' if secret else '（无密钥/未加签）'}")
    text = (
        "## ✅ AlphaGPT 连通性测试\n\n"
        "- 如果你看到这条消息，说明 Webhook 与加签密钥配置正确\n"
        f"- 发送时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    status, body = send(webhook, secret, text)
    print(f"HTTP {status} -> errcode={body.get('errcode')} errmsg={body.get('errmsg')}")
    if body.get("errcode") == 0:
        print("✅ 发送成功，请到钉钉群里确认。")
    else:
        print("❌ 发送失败，对照文件顶部注释排查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
