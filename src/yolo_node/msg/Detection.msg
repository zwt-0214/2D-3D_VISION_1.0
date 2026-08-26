# 单个检测目标的消息类型

# 边界框坐标 (左上角 x, y 和宽高)
float64 x
float64 y
float64 width
float64 height

# 置信度 (0.0 - 1.0)
float64 confidence

# 类别 ID
int32 class_id

# 类别名称
string class_name