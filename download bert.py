from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# 初始化分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 准备输入数据
inputs = tokenizer("Example sentence", return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# 调用模型
outputs = model(input_ids, attention_mask=attention_mask)
last_hidden_state = outputs.last_hidden_state
pooler_output = outputs.pooler_output  # 如果您的模型版本提供此输出

print("Shape of last_hidden_state:", last_hidden_state.shape)
print("Shape of pooler_output:", pooler_output.shape)