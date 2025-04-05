# 示例 1: 明确列出所有标签并要求 0/1
--prompt
"Analyze the provided chest X-ray image. For each of the following 14 findings, state 1 if present and 0 if absent. Output ONLY the list below, replacing [0 or 1] with the appropriate value:
No
Finding: [0 or 1]
Enlarged
Cardiomediastinum: [0 or 1]
Cardiomegaly: [0 or 1]
Lung
Opacity: [0 or 1]
Lung
Lesion: [0 or 1]
Edema: [0 or 1]
Consolidation: [0 or 1]
Pneumonia: [0 or 1]
Atelectasis: [0 or 1]
Pneumothorax: [0 or 1]
Pleural
Effusion: [0 or 1]
Pleural
Other: [0 or 1]
Fracture: [0 or 1]
Support
Devices: [0 or 1]
"

# 示例 2: 更简洁的指令，但仍强调格式
--prompt
"Evaluate this chest X-ray for the 14 CheXpert findings. Report 1 for presence, 0 for absence. Use the format 'Finding Name: Value'. List all 14 findings: No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices."

# 示例 3: 尝试用角色扮演引导
--prompt
"You are a radiology assistant analyzing a chest X-ray. Please report the presence (1) or absence (0) of the following conditions, based *only* on the image provided. Output *only* the list in the format 'Condition: Value'.
Conditions: No
Finding, Enlarged
Cardiomediastinum, Cardiomegaly, Lung
Opacity, Lung
Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural
Effusion, Pleural
Other, Fracture, Support
Devices.
"