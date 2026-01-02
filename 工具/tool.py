import librosa
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoClip, AudioFileClip
import cv2

# ==================== 配置参数 ====================
# 请修改以下路径为你的文件路径
AUDIO_PATH = "./music/tmp68pj4aii.flac"        # 音频文件路径
IMAGE_PATH = "./images/NetaYume_Lumina_3.5_00019_.png"  # 背景图片路径
SONG_NAME = "初戀"                  # 歌曲名称
OUTPUT_PATH = "./videos/初戀.mp4"             # 输出视频路径

# 视频参数
VIDEO_WIDTH = 1920                     # 视频宽度
VIDEO_HEIGHT = 1080                    # 视频高度
FPS = 24                               # 帧率

# 频谱条参数
NUM_BARS = 64                          # 频谱条数量
BAR_WIDTH = 25                         # 频谱条宽度
BAR_COLOR = "#F294D1"                 # 频谱条颜色 RGB
BAR_SCALE = 3.0                        # 频谱条高度缩放系数

# 文字参数
TEXT_SIZE = 200                         # 歌曲名字体大小
TEXT_COLOR = "#F294D1"           # 歌曲名颜色 RGB

# 实时预览
ENABLE_PREVIEW = True                  # 是否开启实时预览（会稍微降低生成速度）
PREVIEW_SCALE = 0.5                    # 预览窗口缩放比例（0.5 = 50%）
# ==================================================

print("正在加载音频文件...")
y, sr = librosa.load(AUDIO_PATH)
duration = librosa.get_duration(y=y, sr=sr)
print(f"音频时长: {duration:.2f} 秒")

print("正在分析音频频谱...")
hop_length = 512
# 计算短时傅里叶变换得到频谱
stft = librosa.stft(y, hop_length=hop_length)
spec = np.abs(stft)

# 将频谱分为指定数量的频段
n_freqs = spec.shape[0]
freqs_per_bar = n_freqs // NUM_BARS
spec_bars = np.array([
    np.mean(spec[i*freqs_per_bar:(i+1)*freqs_per_bar], axis=0) 
    for i in range(NUM_BARS)
])

# 归一化频谱
spec_bars = spec_bars / np.max(spec_bars)

print("正在加载背景图片...")
# 使用 PIL 加载并调整图片大小
bg_image = Image.open(IMAGE_PATH).convert('RGB')
bg_image = bg_image.resize((VIDEO_WIDTH, VIDEO_HEIGHT), Image.Resampling.LANCZOS)

# 尝试加载字体，失败则使用默认字体
print("正在加载字体...")
font = None
font_paths = [
    "C:/Windows/Fonts/simkai.ttf",    # 楷体
]

for font_path in font_paths:
    try:
        font = ImageFont.truetype(font_path, TEXT_SIZE)
        print(f"成功加载字体: {font_path}")
        break
    except:
        continue

if font is None:
    print("警告: 无法加载TrueType字体，使用PIL默认字体（不支持中文）")
    font = ImageFont.load_default(size=TEXT_SIZE)

# 全局变量用于预览
frame_counter = [0]
preview_window_created = [False]

def make_frame(t):
    """生成视频的每一帧 - 使用 PIL 优化性能"""
    # 计算当前时间对应的频谱帧索引
    frame_idx = int(t * sr / hop_length)
    frame_idx = min(frame_idx, spec_bars.shape[1] - 1)
    
    # 获取当前帧的频谱数据
    current_bars = spec_bars[:, frame_idx]
    
    # 复制背景图片
    frame = bg_image.copy()
    draw = ImageDraw.Draw(frame)
    
    # 计算频谱条的位置和高度
    bar_spacing = VIDEO_WIDTH / NUM_BARS
    bar_bottom = int(VIDEO_HEIGHT * 0.85)  # 频谱条底部位置
    
    # 绘制频谱条
    for i in range(NUM_BARS):
        bar_height = int(current_bars[i] * VIDEO_HEIGHT * 0.3 * BAR_SCALE)
        x1 = int(i * bar_spacing + (bar_spacing - BAR_WIDTH) / 2)
        x2 = x1 + BAR_WIDTH
        y1 = bar_bottom - bar_height
        y2 = bar_bottom
        
        # 绘制频谱条
        draw.rectangle([x1, y1, x2, y2], fill=BAR_COLOR, outline=(0, 0, 0))
    
    # 绘制歌曲名（直接在背景上）
    text_bbox = draw.textbbox((0, 0), SONG_NAME, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = int(VIDEO_WIDTH * 0.15)  # 左侧，距离左边缘15%
    text_y = (VIDEO_HEIGHT - text_height) // 3  # 垂直居中偏上
    
    # 绘制文字
    draw.text((text_x, text_y), SONG_NAME, font=font, fill=TEXT_COLOR)
    
    # 转换为 numpy 数组
    frame_array = np.array(frame)
    
    # 实时预览
    if ENABLE_PREVIEW:
        frame_counter[0] += 1
        # 每隔几帧显示一次，避免太频繁
        if frame_counter[0] % 2 == 0:
            # 转换颜色格式 RGB -> BGR（OpenCV使用BGR）
            preview_frame = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            # 缩放预览图像
            if PREVIEW_SCALE != 1.0:
                preview_size = (int(VIDEO_WIDTH * PREVIEW_SCALE), int(VIDEO_HEIGHT * PREVIEW_SCALE))
                preview_frame = cv2.resize(preview_frame, preview_size)
            
            # 在图像上显示进度信息
            progress_text = f"生成中: {t:.1f}s / {duration:.1f}s ({t/duration*100:.1f}%)"
            cv2.putText(preview_frame, progress_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 创建预览窗口（首次）
            if not preview_window_created[0]:
                cv2.namedWindow('视频生成预览 - 按Q退出预览', cv2.WINDOW_NORMAL)
                preview_window_created[0] = True
            
            # 显示图像
            cv2.imshow('视频生成预览 - 按Q退出预览', preview_frame)
            
            # 处理按键事件（按Q关闭预览）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                preview_window_created[0] = False
    
    return frame_array

print("正在生成视频...")
print(f"预计需要生成 {int(duration * FPS)} 帧")
if ENABLE_PREVIEW:
    print("实时预览已开启，生成过程中会显示预览窗口")
    print("提示: 按 Q 键可关闭预览窗口（不影响视频生成）")

video = VideoClip(make_frame, duration=duration)
video = video.with_audio(AudioFileClip(AUDIO_PATH))
video.write_videofile(
    OUTPUT_PATH, 
    fps=FPS, 
    codec='libx264', 
    audio_codec='aac',
    threads=4,  # 使用多线程加速
    preset='ultrafast'  # 使用最快编码预设
)

# 关闭所有预览窗口
if ENABLE_PREVIEW:
    cv2.destroyAllWindows()

print(f"视频生成完成！保存位置: {OUTPUT_PATH}")
