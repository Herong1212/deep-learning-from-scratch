# coding: utf-8
class Man:
    """
    示例类

    这个类是作为示例创建的。
    有关各个方法和属性的详细信息，请参阅各自的定义部分。
    
    """
    def __init__(self, name):
        self.name = name
        print("Initilized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good-bye " + self.name + "!")

m = Man("David")
m.hello()
m.goodbye()