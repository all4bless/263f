import subprocess
import os

# Create an output directory for frames
output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)
#
# # Set parameters for plotting and simulation
# plotStep = 10  # Save a frame every 50 time steps
# dt = 1e-3      # Time step size (s)
#
animation_mp4 = "animation5.mp4"
animation_gif = "animation5.gif"

# Generate MP4
subprocess.run([
    "ffmpeg",
    "-framerate", "20",
    "-i", f"{output_dir}/frame_%06d.png",
    "-c:v", "libx264",
    "-r", "30",
    "-pix_fmt", "yuv420p",
    animation_mp4
], check=True)

# Generate GIF
subprocess.run([
    "ffmpeg",
    "-framerate", "20",
    "-i", f"{output_dir}/frame_%06d.png",
    "-vf", "scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
    "-loop", "0",
    animation_gif
], check=True)


# # 指定包含图片的目录
# directory = r"W:\PyCharmProject\pythonProject\final_project\frames"
#
# # 获取目录中所有文件并排序
# files = sorted(os.listdir(directory))
#
# # 初始化新的文件编号
# new_number = 1
#
# for filename in files:
#     if filename.startswith('frame_'):
#         # 构建新的文件名
#         new_filename = f'frame_{new_number:06d}.png'
#
#         # 重命名文件
#         os.rename(os.path.join(directory, filename),
#                   os.path.join(directory, new_filename))
#
#         # 增加编号
#         new_number += 1

# import matplotlib.pyplot as plt
# import numpy as np
#
# # Data
# configurations = ['3×3', '5×5', '7×7', '9×9', '11×11']
# compressed_ratios = [0.8194, 0.9438, 0.9658, 0.9763, 0.9810]
#
# # Create the plot
# plt.figure(figsize=(8, 6))
# plt.plot(range(len(configurations)), compressed_ratios, 'bo-', linewidth=2, markersize=8)
#
# # Customize the plot
# plt.xlabel('Honeycomb Configuration')
# plt.ylabel('Compressed Ratio')
# plt.title('Compressed Ratio vs Honeycomb Configuration')
# plt.grid(True)
#
# # Set x-axis ticks and labels
# plt.xticks(range(len(configurations)), configurations)
#
# # Set y-axis limits with more space for labels
# plt.ylim(0.75, 1.00)  # Adjusted to provide more space above the highest point
#
# # Add data point labels
# for i, txt in enumerate(compressed_ratios):
#     plt.annotate(f'{txt:.4f}',
#                 (i, compressed_ratios[i]),
#                 textcoords="offset points",
#                 xytext=(0,10),
#                 ha='center')
#
# plt.show()