##  insert images into Excel
import Automatic_Excel_Inserting


'''用于自动读取图片上的地址和时间信息，并修改为图片文件名称。这里提供了两个函数，选择一个即可'''
##  如果你还想要自动获取图片上的时间，运行这个函数
image_folder_path = 'D:\Project\pythonProject\Test'  # 图片所在的文件夹

Automatic_Excel_Inserting.getTimeAuto(image_folder_path)


##  如果不适合自动读取时间，你想要不输入，或者手动输入时间，则运行这个函数
image_folder_path = 'D:\Project\pythonProject\Test'  # 图片所在的文件夹
time = ''  # 比如：'_上班'，'_下班'，等都可以，用于区分时间，注意前面必须有个下划线，并放在单引号中

Automatic_Excel_Inserting.getTimeManually(image_folder_path, time)







'''在运行了上面的函数之后，运行以下函数，可以将图片自动插入到 excel 的目标列中。这里提供了两个函数，选择一个即可'''
##  如果你想要按照文件名称顺序依次插入图片到 Excel的各行，运行这个函数。这个方法实际上不可行，因为名称并不相同，从而排序也不同
excel_folder_path = 'D:\Project\pythonProject\Test'  # 设置excel所在的文件夹路径
image_folder_path = 'D:\Project\pythonProject\Test'  # 设置图片所在的文件夹路径
excel_filename = 'Book1.xlsx'  # excel文件的名称, 必须加上后缀 xlxs
sheet_name = 'Sheet1'  # 想要插入到 excel里的哪一个页面中
starting_row_number = 1  # 第一张图片放置的行号，后续图片按照名称顺序排列
image_column_number = 'B'  # 图片放在哪一列中
img_width = 80  # 图片的宽度
img_height = 80  # 图片的高度

Automatic_Excel_Inserting.insertImageAlphaOrder(excel_folder_path, image_folder_path, excel_filename, sheet_name, starting_row_number, image_column_number,
        img_width, img_height)



## 如果你想以学校名为参照，将图片插入到对应学校所在的那一行，则运行这个函数。但你需要提前定义一个学校名称的对应表
sheet = 'Sheet1'

## 用于上班
excel_folder_path = 'D:\Project\pythonProject\Test'  # 设置excel所在的文件夹
image_folder_path = 'D:\Project\pythonProject\Test'  # 设置图片所在的文件夹
excel_filename = 'Book12.xlsx'  # excel文件的名称, 必须加上后缀 xlxs
sheet_name = sheet  # 想要插入到 excel里的哪一个页面中
school_column_index = 0  # Excel上的学校名称所在列，0指的是第一列， 1指的是第二列，以此类推
image_column_index = 'G'  # 放置图片的列，define the column index to insert
img_width = 60  # 图片的宽度
img_height = 60  # 图片的高度

Automatic_Excel_Inserting.insertImageSameName(excel_folder_path, image_folder_path, excel_filename, sheet_name, image_column_index, school_column_index, img_width,
    img_height, Automatic_Excel_Inserting.school_name_list)
'''特别的，如果运行这个函数时出现如下类似的提示，比如：KeyError: '深圳市宝安区人民政府'，则说明你的列表中没有这个学校，需要将其添加进去'''



## 用于下班
excel_folder_path = 'D:\Project\pythonProject\Test'  # 设置excel所在的文件夹
image_folder_path = 'D:\Project\pythonProject\Test'  # 设置图片所在的文件夹
excel_filename = 'Book12.xlsx'  # excel文件的名称, 必须加上后缀 xlxs
sheet_name = sheet  # 想要插入到 excel里的哪一个页面中
school_column_index = 0  # Excel上的学校名称所在列，0指的是第一列， 1指的是第二列，以此类推
image_column_index = 'G'  # 放置图片的列，define the column index to insert
img_width = 60  # 图片的宽度
img_height = 60  # 图片的高度

Automatic_Excel_Inserting.insertImageSameName(excel_folder_path, image_folder_path, excel_filename, sheet_name, image_column_index, school_column_index, img_width,
    img_height, Automatic_Excel_Inserting.school_name_list)
'''特别的，如果运行这个函数时出现如下类似的提示，比如：KeyError: '深圳市宝安区人民政府'，则说明你的列表中没有这个学校，需要将其添加进去'''

