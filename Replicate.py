import replicate
import os

os.environ["REPLICATE_API_TOKEN"] = "r8_X5rkjHHRqccTNA4yJQhS5EKbEIsFBbT3Sdzge"

output = replicate.run(
    "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    input={
        "debug": False,
        "top_k": 50,
        "top_p": 1,
        "prompt": "病例描述：患者為 45 歲女性，近期經常感到疲勞，有間斷性頭痛，過去兩週出現輕微視力模糊。患者有高血壓病史，但未定期服藥。近期體重略有下降，但未進行特別減肥。請基於這些信息，預測最可能的前五個 ICD-10 代碼。",
        "temperature": 0.5,
        "system_prompt": "你是一位經驗豐富的醫師及疾病分類師，對 ICD-10 的分類系統非常熟悉。你的任務是根據所提供的病例描述，準確預測出最相關的前五個 ICD-10 代碼。你需要依據病例的症狀、病史及其他相關信息來做出判斷。請確保你的回答既精準又具有教育意義，幫助提升對 ICD-10 分類系統的理解。請在回答時列出前五個可能的 ICD-10 代碼，並簡要解釋每個代碼對應的疾病或病情。請以繁體中文回答，並且文本不要隨意換行",
        "max_new_tokens": 500,
        "min_new_tokens": -1
    }
)
# 初始化一个空字符串来累加文本
accumulated_text = ""

# 遍历输出生成器中的每个文本片段
for item in output:
    # 假设每个item是字符串，直接累加（这里没有额外添加空格，假设每个输出片段之间不需要额外的分隔）
    accumulated_text += item

# 打印累加后的完整文本
print(accumulated_text)




