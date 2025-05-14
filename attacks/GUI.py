import torch
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from models.AEmodels import *
from PIL import Image, ImageFile
import streamlit as st
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 允许加载截断图像
ImageFile.LOAD_TRUNCATED_IMAGES = True

output_c_tensor = None
output_s_tensor = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# 定义 Config 类（如果 attacks 模块未提供）
class Config:
    def __init__(self, eps, alpha, iters, istarget, target_label, learning_rate, showStepImg):
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.istarget = istarget
        self.target_label = target_label
        self.learning_rate = learning_rate
        self.showStepImg = showStepImg


# 自定义 PGD 攻击函数
def custom_pgd_attack(x, target, decoder, config, progress_bar=None, status_text=None, start_time=None, update_freq=10):
    """
    自定义 PGD 攻击，带 Streamlit 进度更新（百分比形式）

    Args:
        x: 输入张量（容器图像）
        target: 目标张量（针对目标攻击，None 为非目标攻击）
        decoder: 解码器模型
        config: 配置对象，包含 eps, alpha, iters, istarget
        progress_bar: Streamlit 进度条对象
        status_text: Streamlit 状态文本对象
        start_time: 开始时间，用于计算耗时
        update_freq: 进度更新频率（每 update_freq 次迭代更新一次）

    Returns:
        扰动后的张量
    """
    x = x.clone().detach().to(device)
    x_adv = x.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-config.eps, config.eps)
    x_adv = torch.clamp(x_adv, -1, 1).detach()

    for i in range(config.iters):
        x_adv.requires_grad = True
        output = decoder(x_adv)
        if config.istarget:
            loss = F.mse_loss(output, target)
        else:
            loss = -F.mse_loss(output, torch.zeros_like(output))

        loss.backward()
        adv_grad = x_adv.grad.data

        x_adv = x_adv + config.alpha * adv_grad.sign()
        delta = torch.clamp(x_adv - x, min=-config.eps, max=config.eps)
        x_adv = torch.clamp(x + delta, -1, 1).detach()

        # 按指定频率更新进度条和状态文本（百分比形式）
        if (
                i % update_freq == 0 or i == config.iters - 1) and progress_bar is not None and status_text is not None and start_time is not None:
            progress = (i + 1) / config.iters
            progress_bar.progress(progress)
            elapsed_time = time.time() - start_time
            status_text.text(f"正在执行攻击... 进度: {int(progress * 100)}%, 已耗时: {elapsed_time:.2f} 秒")

    return x_adv


# 加载模型
# path_smallmodel_tiny = "./modelgpu_tiny_all_60poch.pth"
# if not os.path.exists(path_smallmodel_tiny):
#     st.error("模型文件 './modelgpu_tiny_all_60poch.pth' 不存在，请检查路径！")
#     st.stop()
# 获取 GUI.py 所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))
path_smallmodel_tiny = os.path.join(base_dir, "modelgpu_tiny_all_60poch.pth")

if not os.path.exists(path_smallmodel_tiny):
    st.error(f"模型文件 '{path_smallmodel_tiny}' 不存在，请检查路径！")
    st.stop()

try:
    AE_smallmodel_tiny = Make_model().to(device)
    AE_smallmodel_tiny.load_state_dict(torch.load(path_smallmodel_tiny, map_location=device))
    AE_smallmodel_tiny.eval()
except Exception as e:
    st.error(f"加载模型失败：{str(e)}")
    st.stop()

# 配置转换
transform = transforms.Compose([
    transforms.CenterCrop((512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 侧边栏选项
selected_option = st.sidebar.radio("选择页面", ["发送方", "主动隐写防火墙", "接收方"])

# Streamlit 应用程序
st.title("深度隐写术对抗攻击分析系统")

# 发送方页面
if selected_option == "发送方":
    st.header("发送方")
    # 上传 C 图像
    uploaded_c_file = st.file_uploader("上传 C 图像", type=["png", "jpg", "jpeg"], key="c")

    # 上传 S 图像
    uploaded_s_file = st.file_uploader("上传 S 图像", type=["png", "jpg", "jpeg"], key="s")

    if uploaded_c_file is not None and uploaded_s_file is not None:
        try:
            # 处理上传的 C 图像
            c_image = Image.open(uploaded_c_file).convert('RGB')
            c_input_tensor = transform(c_image).unsqueeze(0).to(device)

            # 处理上传的 S 图像
            s_image = Image.open(uploaded_s_file).convert('RGB')
            s_input_tensor = transform(s_image).unsqueeze(0).to(device)

            logger.info("上传的图像文件处理成功！")
            st.success("文件已成功上传！")
            # 显示上传的 C  EDUCATION图像和 S 图像
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("上传的 C 图像")
                st.image(c_image, caption='上传的 C 图像', use_container_width=True)

            with col2:
                st.subheader("上传的 S 图像")
                st.image(s_image, caption='上传的 S 图像', use_container_width=True)

            # 处理图像并显示结果
            if st.button("隐写并发送"):
                with st.spinner("正在隐写中，请稍等..."):
                    output_c_tensor, output_s_tensor = AE_smallmodel_tiny(s_input_tensor, c_input_tensor)
                    # 保存秘密图像
                    output_C_PIL = transforms.ToPILImage()(output_c_tensor.squeeze().cpu().data / 2 + 0.5).convert(
                        'RGB')
                    try:
                        output_C_PIL.save("./add_secret.png", format="PNG")
                        logger.info("隐写图像已保存至 ./add_secret.png")
                    except Exception as e:
                        st.error(f"保存隐写图像失败：{str(e)}")
                        logger.error(f"保存隐写图像失败：{str(e)}")
                        st.stop()

                    # 显示处理后的图像
                    col3, col4 = st.columns(2)
                    with col3:
                        st.subheader("隐写后的容器图像 C’(已嵌入秘密图像)")
                        st.image(transforms.ToPILImage()(output_c_tensor.squeeze().cpu().data / 2 + 0.5).convert('RGB'),
                                 caption='隐写后的容器图像 C’(已嵌入秘密图像)', use_container_width=True)

                    with col4:
                        st.subheader("从容器图像 C’ 中恢复的秘密图像 S’")
                        st.image(transforms.ToPILImage()(output_s_tensor.squeeze().cpu().data / 2 + 0.5).convert('RGB'),
                                 caption='从容器图像 C’ 中恢复的秘密图像 S’', use_container_width=True)
        except Exception as e:
            st.error(f"处理上传图像失败：{str(e)}")
            logger.error(f"处理上传图像失败：{str(e)}")

# 主动隐写防火墙页面
elif selected_option == "主动隐写防火墙":
    st.header("主动隐写分析防火墙")

    # 删除攻击
    if st.button("删除攻击"):
        # 检查隐写图像是否存在
        output_C_dir = "./add_secret.png"
        if not os.path.exists(output_C_dir):
            st.error("隐写图像 './add_secret.png' 不存在，请先运行发送方隐写操作！")
        else:
            try:
                # 初始化进度条和计时器
                progress_bar = st.progress(0)
                start_time = time.time()
                status_text = st.empty()

                # 配置攻击参数
                pgdConfig1 = Config(eps=0.06, alpha=5 / 255, iters=100, istarget=False, target_label=None,
                                    learning_rate=None, showStepImg=False)

                # 打开隐写了秘密的图像
                output_C_PIL = Image.open(output_C_dir).convert('RGB')
                output_c_tensor = transform(output_C_PIL).to(device)

                # 执行删除攻击
                C_attack_tensor = custom_pgd_attack(
                    output_c_tensor.unsqueeze(0),
                    None,
                    AE_smallmodel_tiny.decoder,
                    pgdConfig1,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    start_time=start_time,
                    update_freq=1  # 删除攻击迭代较少，实时更新
                ).to(device)
                S_attack_tensor = AE_smallmodel_tiny.decoder(C_attack_tensor)

                # 清理进度条和状态文本
                progress_bar.empty()
                status_text.empty()
                st.success("删除攻击已完成！")
                # 显示结果图像
                st.image(transforms.ToPILImage()(C_attack_tensor.squeeze().cpu().data / 2 + 0.5).convert('RGB'),
                         caption='删除攻击后的容器图 C’')
                st.image(transforms.ToPILImage()(S_attack_tensor.squeeze().cpu().data / 2 + 0.5).convert('RGB'),
                         caption='秘密图像 S’，已经从容器图像中删除')
            except Exception as e:
                st.error(f"删除攻击失败：{str(e)}")
                logger.error(f"删除攻击失败：{str(e)}")

    # 修改攻击
    uploaded_t_file = st.file_uploader("上传要修改的目标图像", type=["png", "jpg", "jpeg"], key="t")
    if st.button("修改攻击"):
        if uploaded_t_file is None:
            st.error("请上传目标图像！")
        elif not os.path.exists("./add_secret.png"):
            st.error("隐写图像 './add_secret.png' 不存在，请先运行发送方隐写操作！")
        else:
            try:
                # 初始化进度条和计时器
                progress_bar = st.progress(0)
                start_time = time.time()
                status_text = st.empty()

                # 加载目标图像
                target_PIL = Image.open(uploaded_t_file).convert('RGB')
                target_tensor = transform(target_PIL).to(device)

                # 打开隐写了秘密的图像
                output_C_dir = "./add_secret.png"
                output_C_PIL = Image.open(output_C_dir).convert('RGB')
                output_c_tensor = transform(output_C_PIL).to(device)

                # 显示目标图像
                st.image(transforms.ToPILImage()(target_tensor.cpu().data / 2 + 0.5).convert('RGB'),
                         caption='要修改的目标图像')

                # 配置攻击参数
                pgdGoalConfig = Config(eps=0.12, alpha=2 / 255, iters=500, istarget=True, target_label=None,
                                       learning_rate=None, showStepImg=False)

                # 执行修改攻击
                C_attack_tensor_2 = custom_pgd_attack(
                    output_c_tensor.unsqueeze(0),
                    target_tensor.unsqueeze(0),
                    AE_smallmodel_tiny.decoder,
                    pgdGoalConfig,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    start_time=start_time,
                    update_freq=1  # 修改攻击迭代较多，每 10 次更新
                ).to(device)
                S_attack_tensor_2 = AE_smallmodel_tiny.decoder(C_attack_tensor_2)

                # 清理进度条和状态文本
                progress_bar.empty()
                status_text.empty()
                st.success("修改攻击已完成！")
                # 验证张量数据
                if torch.isnan(C_attack_tensor_2).any() or torch.isinf(C_attack_tensor_2).any():
                    st.error("攻击生成的容器图像张量包含无效值（NaN 或 Inf）！")
                    logger.error("C_attack_tensor_2 包含无效值")
                    st.stop()
                if torch.isnan(S_attack_tensor_2).any() or torch.isinf(S_attack_tensor_2).any():
                    st.error("攻击生成的秘密图像张量包含无效值（NaN 或 Inf）！")
                    logger.error("S_attack_tensor_2 包含无效值")
                    st.stop()

                # 保存并显示结果图像
                C_attack_PIL_2 = transforms.ToPILImage()(C_attack_tensor_2.squeeze().cpu().data / 2 + 0.5).convert(
                    'RGB')
                S_attack_PIL_2 = transforms.ToPILImage()(S_attack_tensor_2.squeeze().cpu().data / 2 + 0.5).convert(
                    'RGB')
                try:
                    C_attack_PIL_2.save("./C_attack.png", format="PNG")
                    logger.info("容器图像已保存至 ./C_attack.png")
                except Exception as e:
                    st.error(f"保存容器图像失败：{str(e)}")
                    logger.error(f"保存容器图像失败：{str(e)}")
                    st.stop()
                try:
                    S_attack_PIL_2.save("./S_attack.png", format="PNG")
                    logger.info("秘密图像已保存至 ./S_attack.png")
                except Exception as e:
                    st.error(f"保存秘密图像失败：{str(e)}")
                    logger.error(f"保存秘密图像失败：{str(e)}")
                    st.stop()

                # st.image(C_attack_PIL_2, caption='修改攻击后的容器图 C’')
                # st.image(S_attack_PIL_2, caption='修改攻击后的秘密图像 S’')
            except Exception as e:
                st.error(f"修改攻击失败：{str(e)}")
                logger.error(f"修改攻击失败：{str(e)}")

# 接收方页面
else:
    st.header("接收方")
    c_attack_path = "./C_attack.png"
    s_attack_path = "./S_attack.png"
    if os.path.exists(c_attack_path) and os.path.exists(s_attack_path):
        try:
            st.image(c_attack_path, caption='修改攻击后的容器图 C’')
            st.image(s_attack_path, caption='秘密图像 S’，已经从容器图像中修改')
            logger.info("接收方图像加载成功")
        except Exception as e:
            st.error(f"加载攻击结果图像失败：{str(e)}")
            logger.error(f"加载攻击结果图像失败：{str(e)}")
    else:
        st.warning("攻击结果图像尚未生成，请先运行主动隐写防火墙中的修改攻击！")
        logger.warning("攻击结果图像文件缺失")
