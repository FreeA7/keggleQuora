import os
import re
import logging

class myLogging(object):
	"""	自动日志记录
		需要用createlog进行初始化
	"""
	def __init__(self, name):
		super(myLogging, self).__init__()
		self.name = name
		if os.path.exists('./all_logs'):
			pass
		else:
			os.mkdir('./all_logs')
		if os.path.exists('./all_logs/' + self.name):
			pass
		else:
			os.mkdir('./all_logs/' + self.name)
		self.p = re.compile(name + '[_][0-9]+[.]log')
		self.p_n = re.compile('[0-9]+')

	def getLogNum(self):
		list = os.listdir('./all_logs/' + self.name)
		if len(list) == 0:
			return 1
		num = 0
		for i in list:
			if re.search(self.p, i):
				if int(re.search(self.p_n, i).group()) >= num:
					num = int(re.search(self.p_n, i).group())
		if num == 0:
			return 1
		else:
			return (num + 1)

	def createlog(self):
		logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./all_logs/' + self.name + '/' + self.name + '_' + str(self.getLogNum()) + '.log',
                filemode='w')
		console = logging.StreamHandler()
		console.setLevel(logging.INFO)
		formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
		console.setFormatter(formatter)
		logging.getLogger('').addHandler(console)
		return logging
