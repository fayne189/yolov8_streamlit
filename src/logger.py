import logging
from logging.handlers import RotatingFileHandler

# 注释来自Bing
def get_logger(name, level='INFO', err_file_path='error.log'):
    # 定义一个函数，用于获取一个logger对象
    # 参数name是logger的名称，level是日志的输出级别，err_file_path是错误日志的文件路径
    # 返回值是一个logger对象

    # 配置好logger模块
    handler_error = RotatingFileHandler(err_file_path, maxBytes=1024 * 1024, backupCount=5) # 创建一个文件处理器，用于记录错误日志，文件大小限制为1MB，最多保留5个备份文件
    handler_control = logging.StreamHandler()    # 创建一个流处理器，用于输出日志到控制台
    handler_error.setLevel('ERROR')               # 设置文件处理器的日志级别为ERROR，只记录错误信息
    handler_control.setLevel(level)             # 设置流处理器的日志级别为参数level指定的值，默认为INFO

    selfdef_fmt = '%(asctime)s - %(funcName)s - %(levelname)s - %(message)s' # 定义一个日志格式，包括时间、函数名、日志级别和日志信息
    formatter = logging.Formatter(selfdef_fmt) # 创建一个格式化器，用于将日志格式应用到日志记录中
    handler_error.setFormatter(formatter) # 将格式化器设置给文件处理器
    handler_control.setFormatter(formatter) # 将格式化器设置给流处理器

    
    logger = logging.getLogger(name) # 获取一个名为name的logger对象
    logger.setLevel('DEBUG')     #设置logger对象的日志级别为DEBUG，这样才会把debug以上的输出到控制台或文件中
    
    logger.addHandler(handler_error)    #添加handler
    logger.addHandler(handler_control)
    return logger # 返回logger对象

if __name__ == "__main__":
    logger = get_logger(__name__, level='DEBUG')
    logger.debug(123)
    logger.info("xxx %s" % 'info,一般的信息输出')
    logger.warning('waring，用来用来打印警告信息')
    logger.error('error，一般用来打印一些错误信息')
    logger.critical('critical，用来打印一些致命的错误信息，等级最高')