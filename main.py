import random
import math
#name:angrypartyヾ(≧▽≦*)o
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

嚴重程度 = list(range(-32, 33))

array = [
    [[0, 1], [1, 0]],
    [[1, 0], [0, 1]]
]

murmur = True

def m(txt, self):
    ch = {
        Eyes: "眼睛",
        Species: "生物",
        Env: "環境",
        Weight: "權重",
        Output:'輸出包裝'
    }
    if murmur:
        print(f"#{ch[type(self)]}{self.id}:{txt}")

def to_flag(val):
    step = 1/32
    idx = int(val/step)
    idx += 32
    idx = max(0, min(idx, len(嚴重程度) - 1))
    return idx**2

def loss_fn(out, target):
    if len(out) != len(target):
        raise Exception(f"Loss 維度不匹配: 輸出{len(out)} vs 目標{len(target)}")
    loss = []
    for o, t in zip(out, target):
        loss.append((o, to_flag(o.v - t)))
    return loss

def h(flag):
    return 1 if flag > 0 else 0

class Weight():
    chance = [0.1, 0.4, 0.4, 0.4, 0.1]
    
    def __init__(self, idx):
        self.id = 0
        self.idx = idx
        self._立場 = h(idx)
        self.立場 = 1 if idx > 0 else -1
        self.__str__ = self.__repr__
        self._idx = idx + 2
        
    @property
    def name(self):
        ch = ["極端悲觀", "悲觀", "中立", "樂觀", "極端樂觀"]
        return ch[self._idx]

    def 改變(self, force):
        original = self._idx
        self._idx += force
        self._idx = max(0, min(4, self._idx))
        
        if self._idx != original:
            self.idx = self._idx - 2
            return True
        else:
            return False

    @staticmethod
    def sit():
        sit = [-2, -1, 0, 1, 2]
        return random.choices([Weight(n) for n in sit], [0.1, 0.4, 0.4, 0.4, 0.1], k=1)[0]

    def __repr__(self):
        return f'個性:{self.name}'
        
    def update(self):
        idx = self.idx
        self._立場 = h(idx)
        self.立場 = 1 if idx > 0 else -1
        self._idx = idx + 2

    def handle(self, val):
        opt = [-1, -val, 0, val, 1]
        return opt[self._idx]

class Output():
    id=0
    def __init__(self, val, source):
        self.v = val
        self.s = source
        self.__str__ = self.__repr__
        self.id=Output.id
        Output.id+=1

    def blame(self, flag, w=None):
        if w is None:
            w = Weight(1)
        if w.idx < 0:
            if self.s:
                m(f'來源為{w} 因此我得進行責怪反面化',self)
                self.s.blame(-flag)
        else:
            if self.s:
                self.s.blame(flag)

    def __repr__(self):
        return f"Out:{self.v} from {self.s}"

class Eyes():
    def __init__(self):
        self.w = Weight.sit()
        self.忍耐上限 = random.randint(64, 128)
        self.比較閾值 = 0.5
        self._目前怒氣 = 0
        self.id = random.randint(0, int(10e7))
        self.處理的貨物 = None
        self.__str__ = self.__repr__

    def __repr__(self):
        return f'<eyes {self.id} with {self.w}>'
    
    def handle(self, item):
        self.處理的貨物 = item
        m(f"得到{item}", self)
        if isinstance(item, Output):
            item = item.v
        val = item
        out_v = self.w.handle(val)
        m(f"因為我是{self.w}，我最後輸出{out_v}", self)
        return out_v
    
    @property
    def 目前怒氣(self):
        return self._目前怒氣
    
    @目前怒氣.setter
    def 目前怒氣(self, v):
        self._目前怒氣 = v

    def 反應(self):
        if isinstance(self.處理的貨物, Output):
            self.處理的貨物.s.反應()
            
        if abs(self.目前怒氣) > self.忍耐上限:
            excess = abs(self.目前怒氣) - self.忍耐上限
            step = 1 + int(excess / 64)
            
            pos = step if self.目前怒氣 > 0 else -step
            
            old_name = self.w.name
            changed = self.w.改變(pos)
            self.w.update()
            
            if changed:
                m(f'翻桌! 怒氣溢出{excess}，我從個性:{old_name}變為{self.w}', self)
            else:
                m(f'翻桌! 但我已經是最極端了 ({self.w})，變得更敏感', self)
                self.忍耐上限-=10
                
                
            self.目前怒氣 = 0
            return 1
        return 0

    def blame(self, flag):
        fac1 = h(flag)
        val = self.處理的貨物
        if isinstance(val, Output):
            val = val.v
        fac2 = int(val > self.比較閾值)
        ch = ["小於", "大於"]
        fac3 = self.w._立場
        ch2 = ["阻攔", "放行"]
        罪 = array[fac1][fac2][fac3]
        hk = ["無罪", "有罪"]
        m(f"環境說{ch[fac1]},我處理的值比閾值{ch[fac2]},我的動作是{ch2[fac3]} 所以說我{hk[罪]},怒氣加{flag}", self)
        if 罪:
            self.目前怒氣 += flag
        m(f"我目前的怒氣為［{self.目前怒氣}/{self.忍耐上限}]", self)
        if isinstance(self.處理的貨物, Output):
            self.處理的貨物.blame(flag, w=self.w) 

class Species():
    id = 0
    def __init__(self, num_eyes, id=None):
        if id is None:
            id = random.randint(0, int(10e7))
        
        self.eyes = []
        self.id = id
        self.id = Species.id
        Species.id += 1
        self.rate = 0
        self.__str__ = self.__repr__
        for n in range(num_eyes):
            e = Eyes()
            e.id = f"{self.id}-{n}"
            self.eyes.append(e)

    def output(self, input_val):
        
       
        if not isinstance(input_val, list):
            input_val = [input_val] * len(self.eyes)
        
       
        if len(input_val) != len(self.eyes):
             raise Exception(f"Species Output 維度不匹配! 眼睛有{len(self.eyes)}顆，但輸入了{len(input_val)}個值")

        s = 0
        for n, ns in zip(input_val, self.eyes):
            s += ns.handle(n)
        ks = s / len(self.eyes)
        
        for n in self.eyes:
            n.比較閾值 = ks
        
        s = math.tanh(s*2)
        m(f"輸出{s},閾值更新為{ks}", self)
        return Output(s, self)

    def blame(self, flag):
        
        for n in self.eyes:
            n.blame(flag)
            
    def 反應(self):
        s = 0
        for n in self.eyes:
            s += n.反應()
        self.rate = s / len(self.eyes)
        
    def __repr__(self):
        return f'生物{self.id} with eyes:{"|".join([n.__str__() for n in self.eyes])}'

class Env():
    id=0
    def __init__(self, *shape):
        if len(shape) != 2:
            raise Exception("目前只開放二維")
        Env.id+=1
        self.id=Env.id
        dim1 = shape[0]
        dim2 = shape[1]
        self.dim = dim2
        self.species = []
        for n in range(dim2):
            self.species.append(Species(dim1, id=n))
        self.species.sort(key=lambda x: x.rate)
    
    def handle(self, val):
        out = []
        
        if len(self.species[0].eyes) != len(val):
            raise Exception(f'Env Handle 維度不匹配! 輸入元素{val}，但生物有{len(self.species[0].eyes)}')
            
        for n in self.species:
            opt=n.output(val)
            m(f'收集到來自{n}的輸出{opt}',self)
            out.append(opt)
            
        m(f'輸出{out}',self)
        return out
        
    def __call__(self, val):
        return self.handle(val)
        
    def kill(self):
        num = int(self.dim * 5 / 100)
        pn = int(self.dim * 2 / 100)
        good = self.species[-pn:]
        sur = [random.choice(good) for n in range(pn)] if good else []
        die = self.species[:num]
        if die and good:
            for n in die:
                if n in self.species:
                    self.species.remove(n)
            sur.sort(key=lambda x: x.rate)
            self.species += sur

    def blame(self, loss):
        
        if len(loss) != len(self.species):
            raise Exception(f'Env Blame 維度不匹配! 輸入loss長度{len(loss)}，但生物有{len(self.species)}')

        for n in range(len(loss)):
            if n < len(self.species):
                m(f'問責{n}',self)
                self.species[n].blame(loss[n])

    def 反應(self):
        for n in self.species:
            m(f'觸發{n}',self)
            n.反應()



def train_fn(data, target, epoch):
    for n in range(epoch):
        print(f"[{n+1}/{epoch}]")
        for i, (d, t) in enumerate(zip(data, target)):
            x = d
            for env in envs:
                x = env.handle(x)
            loss = loss_fn(x, t)
            for l in loss:
                obj, flg = l
                obj.blame(-flg)
            for l in loss:
                obj, flg = l
                if obj.s:
                    obj.s.反應()

if __name__=='__main__':
    
    k = Env(2,2) 
    g = Env(2, 2) 
    
    for n in range(5):
        
        x = k([2, 6]) 
        
        
       
        x = g(x) 
        
        
        
        g.blame([32, -32]) 
        
        g.反應()
        print(f'{n+1}次輸出:{x}') 
