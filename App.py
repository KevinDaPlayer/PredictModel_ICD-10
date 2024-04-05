# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import replicate
import os

app = Flask(__name__)

os.environ["REPLICATE_API_TOKEN"] = "r8_X5rkjHHRqccTNA4yJQhS5EKbEIsFBbT3Sdzge"

@app.route('/predict_icd10',methods=['POST'])
def predict_icd10():
    data = request.json  # 获取整个请求体的 JSON 字典
    prompt_text = data.get('prompt',)
    if not prompt_text:
        return jsonify("NO INPUT QUERY")

    output = replicate.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "debug": False,
            "top_k": 50,
            "top_p": 1,
            "prompt": "病例描述" + prompt_text + "請基於這些信息，預測最可能的前五個 ICD-10 代碼。",
            "temperature": 0.5,
            "system_prompt": "你是一位經驗豐富的醫師及疾病分類師，對 ICD-10 的分類系統非常熟悉。你的任務是根據所提供的病例描述，準確預測出最相關的前五個 ICD-10 代碼。你需要依據病例的症狀、病史及其他相關信息來做出判斷。請確保你的回答既精準又具有教育意義，幫助提升對 ICD-10 分類系統的理解。請在回答時列出前五個可能的 ICD-10 代碼，並簡要解釋每個代碼對應的疾病或病情。請以繁體中文回答，並且文本不要隨意換行",
            "max_new_tokens": 500,
            "min_new_tokens": -1
        }
    )
    accumulated_text = ""
    for item in output:
        accumulated_text += item
    print(accumulated_text)

    return jsonify({"result" : accumulated_text})

if __name__ == '__main__':
    app.run(debug=True)
