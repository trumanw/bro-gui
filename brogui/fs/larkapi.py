import os
import logging
import numpy as np
import pandas as pd
from larksuiteoapi import Config, DOMAIN_FEISHU, DefaultLogger, MemoryStore, LEVEL_ERROR, ACCESS_TOKEN_TYPE_TENANT
from larksuiteoapi.api import Request

class LarkSheetSession:
    def __init__(
        self, 
        access_token_type=ACCESS_TOKEN_TYPE_TENANT,
        lark_config=None
    ):
        self._access_token_type = access_token_type
        if lark_config is not None:
            self._config = lark_config
        else:
            # load app_id and app_secret from environment variables
            app_settings = Config.new_internal_app_settings_from_env()
            self._config = Config(
                DOMAIN_FEISHU, 
                app_settings, 
                logger=DefaultLogger(), 
                log_level=LEVEL_ERROR, 
                store=MemoryStore())

    @property
    def access_token_type(self):
        return self._access_token_type or ACCESS_TOKEN_TYPE_TENANT

    @property
    def config(self):
        return self._config

    def load_trials_from_remote(self, sheet_token, sheet_index, sheet_col_range="A:J"):
        http_method = "GET"
        meta_uri =  f"https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{sheet_token}/metainfo"

        req = Request(meta_uri, http_method, self.access_token_type, None, request_opts=None)
        resp = req.do(self.config)

        if 0 == resp.code:
            sheet_id = resp.data["sheets"][sheet_index]["sheetId"]
            sheet_range = f"{sheet_id}!{sheet_col_range}"
            data_uri = f"https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{sheet_token}/values/{sheet_range}?valueRenderOption=FormattedValue"

            req = Request(data_uri, http_method, self.access_token_type, None, request_opts=None)
            resp = req.do(self.config)

            if 0 == resp.code:
                data_cols = resp.data['valueRange']['values'][0]
                if 1 == len(resp.data['valueRange']['values']):
                    data_array = None
                else:
                    data_array = np.array(resp.data['valueRange']['values'][1:])
                df = pd.DataFrame(data_array, columns=data_cols)
                return(df, sheet_id, resp.code, None)
            else:
                return(None, None, resp.code, resp.error)
        else:
            return(None, None, resp.code, resp.error)

    def save_trials_to_remote(self, sheet_token, sheet_id, data_array, sheet_col_range="A:J"):
        http_method = "PUT"
        save_uri = f"https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{sheet_token}/values"
        modified_sheet_range = sheet_col_range.split(":")[0]+"2"+":"+sheet_col_range.split(":")[1]
        sheet_range = f"{sheet_id}!{modified_sheet_range}{len(data_array) + 1}"
        request_body = {
            "valueRange": {
                "range": sheet_range,
                "values": data_array
            }
        }

        req = Request(save_uri, http_method, self.access_token_type, request_body, request_opts=None)
        resp = req.do(self.config)

        if 0 == resp.code:
            # stylish the sheet cells
            sheet_range = f"{sheet_id}!A2:I{len(data_array) + 1}"
            stylish_uri = f"https://open.feishu.cn/open-apis/sheets/v2/spreadsheets/{sheet_token}/style"
            request_body = {
                "appendStyle": {
                    "range": sheet_range,
                    "style": {
                        "hAlign": 1,
                        "vAlign": 1,
                        "borderType": "FULL_BORDER"
                    }
                }
            }
            req = Request(stylish_uri, http_method, self.access_token_type, request_body, request_opts=None)
            resp = req.do(self.config)

            return(True, resp.code, None)
        else:
            return(False, resp.code, resp.error)