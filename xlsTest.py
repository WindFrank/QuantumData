import openpyxl  # openpyxl引入模块


def write_to_excel(path: str, sheetStr, info, data):
    #     实例化一个workbook对象
    workbook = openpyxl.Workbook()
    # 激活一个sheet
    sheet = workbook.active
    # 为sheet设置一个title
    sheet.title = sheetStr

    # 添加表头（不需要表头可以不用加）
    #data.insert(0, list(info))
    # 开始遍历数组
    for row_index, row_item in enumerate(data):

        for col_index, col_item in enumerate(row_item):
            # 写入
            sheet.cell(row=row_index + 1, column=col_index + 1, value=col_item)

    # 写入excel文件 如果path路径的文件不存在那么就会自动创建
    workbook.save(path)
    print('写入成功')


if __name__ == '__main__':
    # 数据结构1 path 文件的路径
    path = f'demo1.xlsx'
    # 数据结构1Excel 中sheet 的名字
    sheetStr = '这是数据结构1'

    info = ['name', 'age', 'address']
    # 数据结构1数据
    writeData = [['John Brown', 18, 'New York No. 1 Lake Park']]

    # 执行
    write_to_excel(path, sheetStr, info, writeData)