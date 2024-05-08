# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import replicate
import os

app = Flask(__name__)

os.environ["REPLICATE_API_TOKEN"] = "r8_FE7Ff4mYz5di3suUx7WS4nmueyrvYtT2L12rl"

@app.route('/predict_icd10',methods=['POST'])
def predict_icd10():
    data = request.json  # 获取整个请求体的 JSON 字典
    prompt_text = data.get('prompt',)
    if not prompt_text:
        return jsonify("NO INPUT QUERY")

    output = replicate.run(
         "meta/meta-llama-3-70b-instruct",
    input={
        "debug": False,
        "top_k": 50,
        "top_p": 1,
        "prompt": "病例描述" + prompt_text + "請基於這些信息，預測最可能的前五個 ICD-10 代碼。",
        "temperature": 0.5,
        "system_prompt": "你是一位經驗豐富的醫師及疾病分類師，對 ICD-10 的分類系統非常熟悉。你的任務是根據所提供的病例描述，準確預測出最相關的前五個 ICD-10 代碼。請將預測的代碼依序列出，每個代碼之間以分號分隔，不要包含額外的空格或換行。格式範例：1. K35.8 :疾病解釋;2. I10: 疾病解釋;3. E11.9 :疾病解釋;4. J45.909 :疾病解釋;5. M54.2 :疾病解釋。直接有icd10 code以及explanation for each code就好，解釋可以完整一點，無須再有一次簡短的回答。你需要依據病例的症狀、病史及其他相關信息來做出判斷。請確保你的回答既精準又具有教育意義，幫助提升對 ICD-10 分類系統的理解。",
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


   # input={
   #       "top_k": 50,
   #       "top_p": 0.9,
   #       "prompt": "病例描述" + prompt_text + "請基於這些信息，預測最可能的前五個 ICD-10 代碼。",
   #       "max_tokens": 512,
   #       "min_tokens": 0,
   #       "temperature": 0.6,
   #       "prompt_template": "你是一位經驗豐富的醫師及疾病分類師，對 ICD-10 的分類系統非常熟悉。你的任務是根據所提供的病例描述，準確預測出最相關的前五個 ICD-10 代碼，每個代碼間以;隔開。你需要依據病例的症狀、病史及其他相關信息來做出判斷。請確保你的回答既精準又具有教育意義，幫助提升對 ICD-10 分類系統的理解。請在回答時直接列出前五個可能的 ICD-10 代碼，並簡要解釋每個代碼對應的疾病或病情。文本不要隨意換行",
   #       "presence_penalty": 1.15,
   #       "frequency_penalty": 0.2
   #   }
