from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import PIL
import copy

# 绘制采样点
def only_paint_samplePoint(args, test_loader, patch_location, samplePoint_inPatch, ori_imgNum,
                           deformable_num, show_img):
    """
    patch_location: 取用图片中的哪一个patch，哪一行哪一列，用于计算patch的编号，索引从（1，1）开始，到（8，8）最大
    samplePoint_inPatch: 取出的patch中选择可视化哪个位置的采样点索引从（1，1）开始
    ori_imgNum: 选择哪一行图进行可视化，索引从0开始
    """
    offsets1 = args.paint_lists[5][ori_imgNum]  # size=(81,18,8,8)
    offsets2 = args.paint_lists[6][ori_imgNum]  # size=(81,18,8,8)
    patch_num = (patch_location[0] - 1) * 9 + patch_location[1]
    patch_img = to_pil_image(test_loader.dataset.paint_patches[ori_imgNum][0][patch_num-1].detach())

    # 取出选定位置，选定注意点的偏置数据
    sample_x, sample_y = samplePoint_inPatch[0], samplePoint_inPatch[1]
    offset1 = offsets1[patch_num, :, sample_x - 1, sample_y - 1]
    offset2 = offsets2[patch_num, :, sample_x - 1, sample_y - 1] if deformable_num == 2 else None
    ori_img = args.ori_imgs[ori_imgNum]  # 取出对应原图


    P0 = np.array([[i, j] for i in range(3) for j in range(3)])
    offset1 = offset1.reshape(9, 2)
    offset2 = offset2.reshape(9, 2) if deformable_num == 2 else None
    P1 = P0 + offset1
    P2 = np.array([p1 + offset2 for p1 in P1]) if deformable_num == 2 else None
    plt.figure(figsize=(8, 8))
    # 在patch图上绘制
    ori_draw = ImageDraw.Draw(ori_img)
    ori_draw.rectangle([((patch_location[1]-1)*args.patch_size, (patch_location[0]-1)*args.patch_size),
                        ((patch_location[1])*args.patch_size, (patch_location[0])*args.patch_size)], fill=None, outline="red", width=2)
    draw = ImageDraw.Draw(patch_img)
    # 画到原图上相比特征图扩张4倍
    draw.rectangle([sample_x * 4 - 1, sample_y * 4 - 1, sample_x * 4, sample_y * 4], fill='green')
    # draw.point((sample_x, sample_y), fill='red')

    for point in P1:
        plt.scatter(*point, c='blue')
        plt.text(*point, 'L1', color='blue')
        # paint_x = point[0] + paint_x - 1
        # paint_y = point[1]+paint_y-1
        paint_x = point[0] + sample_x - 1
        paint_y = point[1] + sample_y - 1
        if paint_x < 0 or paint_x > 7 or paint_y < 0 or paint_y > 7:
            continue
        # draw.rectangle([(paint_x * 4) - 1, paint_y * 4 - 1, paint_x, paint_y], fill='blue')
        draw.point(((paint_x * 4) - 1, paint_y * 4 - 1), fill='blue')

    if deformable_num == 2:
        for points in P2:
            for point in points:
                paint_x = point[0] + sample_x - 1
                paint_y = point[1] + sample_y - 1
                plt.scatter(*point, c='red')
                plt.text(*point, 'L2', color='red')
                if paint_x < 0 or paint_x > 7 or paint_y < 0 or paint_y > 7:
                    continue
                draw.point(((paint_x * 4) - 1, paint_y * 4 - 1), fill='red')

    plt.grid(True)
    draw.rectangle([sample_x * 4 - 1, sample_y * 4 - 1, sample_x * 4, sample_y * 4], fill='yellow')
    choose=1
    index=1
    image_path = f"./visualize_data/samplePoint_image/{choose}_{index + 1}_heatmap.jpg"
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    patch_img.save("./visualize_data/samplePoint_image/patch_img.jpg")
    ori_img.save("./visualize_data/samplePoint_image/ori_img.jpg")
    if show_img:
        patch_img.show()
        ori_img.show()
        plt.show()


# 选择哪个层的数据进行绘制，用choose参数选择
def choose_paint(paint_lists, args, test_loader, heatmap_dir, choose=0):
    for index, mean_paint_mat in enumerate(paint_lists[choose]):
        # paint_heatmap(mean_paint_mat, index, choose)
        draw_heatmap(args.ori_imgs[index], mean_paint_mat, index, args, test_loader, choose,
                     heatmap_dir)
    # draw_heatmap(args.ori_imgs[0], paint_lists[choose][0], 0, args, test_loader, choose)

    # offsets1 = args.paint_lists[5][0]  # (81,18,8,8)
    # offsets2 = args.paint_lists[6][0]
    # patch_point = (5, 5)
    # patch_num = (patch_point[0] - 1) * 8 + patch_point[1]
    # patch_x, patch_y = 3, 3 # 4, 6
    # patch_img = to_pil_image(test_loader.dataset.paint_patches[0][0][patch_num].detach())
    # draw_samplePoint(args.ori_imgs[0], patch_img, offsets1[patch_num, :, patch_x - 1, patch_y - 1],
    #                  offsets2[patch_num, :, patch_x - 1, patch_y - 1],
    #                  patch_x, patch_y, patch_num)


# 绘制heatmap
def draw_heatmap(ori_img, heatmap, index, args, test_loader, choose=0, save_dir="./visualize_data/heatmap"):
    """
    ori_img: PIL type
    heatmap: ndarray type
    """
    image_path = save_dir + f"/{choose}_{index + 1}_heatmap.jpg"

    normalized_matrix = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    fig, ax = plt.subplots()
    ax.axis('off')  # removes the axis markers
    ax.imshow(ori_img)


    # img = copy.deepcopy(ori_img)
    # # 创建一个可以在图像上绘图的对象
    # draw = ImageDraw.Draw(img)
    # # 定义网格大小
    # grid_size = 9
    # width, height = img.size
    # x_step = width / grid_size
    # y_step = height / grid_size
    # # 绘制红色直线
    # for i in range(grid_size + 1):
    #     # 绘制垂直线
    #     x = i * x_step
    #     draw.line([(x, 0), (x, height)], fill='red', width=1)
    #     # 绘制水平线
    #     y = i * y_step
    #     draw.line([(0, y), (width, y)], fill='red', width=1)
    # # 显示图像
    # img.show()

    overlay = to_pil_image(normalized_matrix, mode='F').resize((ori_img.size[0], ori_img.size[1]), resample=PIL.Image.Resampling.BICUBIC)
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    ax.imshow(overlay, alpha=0.4, interpolation='nearest')
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    # plt.show()



#======================
# 一些暂存的代码
#======================
# mean_paint_lists = [[np.mean(paint_mat, axis=1).reshape(9, 9) for paint_mat in paint_list] for paint_list in
#                     args.paint_lists[0:5]]
# 画出n组数据，数据的索引在下面这个列表中，choose_paint()负责将每组数据中的m张图画出来
# for choose_index in [0, 2]:
#     choose_paint(mean_paint_lists, args, test_loader, choose=choose_index)
# choose_paint(mean_paint_lists, choose=2)

# 绘制采样点
# def draw_samplePoint(ori_img, patch_img, offset1, offset2, sample_x, sample_y, patch_num):
#     P0 = np.array([[i, j] for i in range(3) for j in range(3)])
#     offset1 = offset1.reshape(9, 2)
#     offset2 = offset2.reshape(9, 2)
#     P1 = P0 + offset1
#     P2 = np.array([p1 + offset2 for p1 in P1])
#     plt.figure(figsize=(8, 8))
#     # 在patch图上绘制
#     ori_draw = ImageDraw.Draw(ori_img)
#     draw = ImageDraw.Draw(patch_img)
#     # 画到原图上相比特征图扩张4倍
#     draw.rectangle([sample_x * 4 - 1, sample_y * 4 - 1, sample_x * 4, sample_y * 4], fill='green')
#     # draw.point((sample_x, sample_y), fill='red')
#
#     for point in P1:
#         plt.scatter(*point, c='blue')
#         plt.text(*point, 'L1', color='blue')
#         # paint_x = point[0] + paint_x - 1
#         # paint_y = point[1]+paint_y-1
#         paint_x = point[0] + sample_x - 1
#         paint_y = point[1] + sample_y - 1
#         if paint_x < 0 or paint_x > 7 or paint_y < 0 or paint_y > 7:
#             continue
#         # draw.rectangle([(paint_x * 4) - 1, paint_y * 4 - 1, paint_x, paint_y], fill='blue')
#         draw.point(((paint_x * 4) - 1, paint_y * 4 - 1), fill='blue')
#
#     for points in P2:
#         for point in points:
#             paint_x = point[0]+sample_x-1
#             paint_y = point[1]+sample_y-1
#             plt.scatter(*point, c='red')
#             plt.text(*point, 'L2', color='red')
#             if paint_x < 0 or paint_x > 7 or paint_y < 0 or paint_y > 7:
#                 continue
#             draw.point(((paint_x * 4) - 1, paint_y * 4 - 1), fill='red')
#
#     plt.grid(True)
#     draw.rectangle([sample_x * 4 - 1, sample_y * 4 - 1, sample_x * 4, sample_y * 4], fill='green')
#     patch_img.show()
#     plt.show()