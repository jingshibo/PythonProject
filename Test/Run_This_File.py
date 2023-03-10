##  insert images into Excel
from Test import Automatic_Excel_Inserting

## 设置存储参数，这三个参数每次使用时都要进行设置。上面两个是图片文件夹的地址，下面一个是excel文件的名称。
shangban_image_folder_path = '/Test'  # 上班图片所在的文件夹
xiaban_image_folder_path = '/Test'  # 下班图片所在的文件夹
excel_folder_path = '/Test'  # 设置excel所在的文件夹
excel_filename = 'Book2.xlsx'  # excel文件的名称, 必须加上后缀 .xlxs


# 上面设置好之后。通常直接依次运行下面所有模块即可，不需要改动任何地方
## 设置更多参数，这里的参数只在初次使用时设置一次，之后很长时间应该都不用设置了，直接运行即可
sheet_name = 'Sheet1'  # excel内部页面的名称
school_column_index = 1  # Excel上的学校名称所在列，0指的是第一列， 1指的是第二列，以此类推
shangban_image_column_index = 'H'  # 放置上班图片的列
xiaban_image_column_index = 'I'  # 放置下班图片的列
img_width = 60  # 图片的宽度
img_height = 60  # 图片的高度


##  读取上班图片的地址，并修改为图片文件名称。
time = ''  # time的值用于在文件名后加入更多信息。比如：'_上班'，'_下班'等。注意前面必须有个下划线，并放在单引号中
Automatic_Excel_Inserting.getTimeManually(shangban_image_folder_path, time)
# 如果有两个图片被识别为了同一个地址，则会在地址后面加上_1，你需要去确认是怎么回事，比如是不是把上下班的图片放到了一个文件夹中。
# 如果你还想要自动获取图片上的时间，运行这个函数
# Automatic_Excel_Inserting.getTimeAuto(shangban_image_folder_path)


## 将上班图片自动插入到 excel 的目标列中。
Automatic_Excel_Inserting.insertImageSameName(excel_folder_path, shangban_image_folder_path, excel_filename, sheet_name, shangban_image_column_index, school_column_index, img_width,
    img_height, Automatic_Excel_Inserting.school_name_list)





##  读取下班图片的地址，并修改为图片文件名称。
time = ''  # time的值用于在文件名后加入更多信息。比如：'_上班'，'_下班'等。注意前面必须有个下划线，并放在单引号中
Automatic_Excel_Inserting.getTimeManually(xiaban_image_folder_path, time)
# 如果有两个图片被识别为了同一个地址，则会在地址后面加上_1，你需要去确认是怎么回事，比如是不是把上下班的图片放到了一个文件夹中。
# 如果你还想要自动获取图片上的时间，运行这个函数
# Automatic_Excel_Inserting.getTimeAuto(xiaban_image_folder_path)


## 将下班图片自动插入到 excel 的目标列中。
Automatic_Excel_Inserting.insertImageSameName(excel_folder_path, xiaban_image_folder_path, excel_filename, sheet_name, xiaban_image_column_index, school_column_index, img_width,
    img_height, Automatic_Excel_Inserting.school_name_list)






# ##  如果你想要按照文件名称顺序依次插入图片到 Excel的各行，运行这个函数。这个方法实际上不可行，因为名称并不相同，从而排序也不同
# excel_folder_path = 'D:\Project\pythonProject\Test'  # 设置excel所在的文件夹路径
# image_folder_path = 'D:\Project\pythonProject\Test'  # 设置图片所在的文件夹路径
# excel_filename = 'Book1.xlsx'  # excel文件的名称, 必须加上后缀 xlxs
# sheet_name = 'Sheet1'  # 想要插入到 excel里的哪一个页面中
# starting_row_number = 1  # 第一张图片放置的行号，后续图片按照名称顺序排列
# image_column_number = 'B'  # 图片放在哪一列中
# img_width = 80  # 图片的宽度
# img_height = 80  # 图片的高度
#
# Automatic_Excel_Inserting.insertImageAlphaOrder(excel_folder_path, image_folder_path, excel_filename, sheet_name, starting_row_number, image_column_number,
#         img_width, img_height)