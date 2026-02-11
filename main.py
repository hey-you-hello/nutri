import random
import math
from collections import Counter
import time




嚴重程度 = list(range(-32, 33))
murmur = False

def m(txt, self):
    
    ch = {
        Eyes: "眼睛",
        Species: "生物",
        Env: "環境",
        Weight: "權重",
        Output: '輸出包裝'
    }
    if murmur:
        try:
            print(f"#{ch[type(self)]}{self.id}:{txt}")
        except KeyError:
            print(f"#未知類型{getattr(self, 'id', '??')}:{txt}")

def to_flag(val):
    step = 1/32
    idx = int(val/step)
    
    return idx

def loss_fn(out, target):
    if len(out) != len(target):
        raise Exception(f"Loss 維度不匹配: 輸出{len(out)} vs 目標{len(target)}")
    loss = []
    for o, t in zip(out, target):
        loss.append(to_flag(o.v-t))
    return loss

def h(flag):
    return 1 if flag > 0 else 0
def w(opt):
    return 1 if opt>= 0 else -1
class Weight():
    chance = [0.1, 0.4, 0.4, 0.4, 0.1]
    
    def __init__(self, idx):
        self.id = 0
        self.idx = idx
        self._立場 = h(idx)
        self._idx=idx+2
        
        
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
    @property
    def 立場(self):
        opt = [-1, -1, 0, 1, 1]
        return opt[self._idx]


    def update(self):
        idx = self.idx
       
        self._idx = idx + 2

    def handle(self, val):
        # 這裡根據目前的狀態決定如何處理數值
        opt = [-1, -val, 0, val, 1]
        return opt[self._idx]
def analy(envs):
    l=[]
    for env in envs:
        for n in env.species:
            for r in n.eyes:
                l.append(r.w.name)
    return Counter(l)

class Output():
    id = 0
    def __init__(self, val, source):
        self.v = val
        self.s = source
        self.id = Output.id
        Output.id += 1

    def blame(self, flag, w=None):
        if w is None:
            w = Weight(1)
        if w.idx < 0:
            if self.s:
                m(f'來源為{w} 因此我得進行責怪反面化', self)
                self.s.blame(-flag)
        else:
            if self.s:
                self.s.blame(flag)

    def __repr__(self):
        return f"Out:{self.v} from {self.s}"

class Eyes():
    def __init__(self):
        self.w = Weight.sit()
        self.忍耐上限 = random.randint(32,128)
        self.比較閾值 = 0.5
        self._目前怒氣 = 0
        self.id = random.randint(0, int(10e7))
        self.處理的貨物 = None

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
        
        return max(-256,min(self._目前怒氣,256))
    
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
            
            
            if changed:
                m(f'翻桌! 怒氣溢出{excess}，我從個性:{old_name}變為{self.w}', self)
                self.w.update()
            else:
                m(f'翻桌! 但我已經是最極端了 ({self.w})，變得更敏感', self)
                
                
            self.目前怒氣 = 0
            return 1
        return 0

    def blame(self, flag):
        val = self.處理的貨物
        if isinstance(val, Output):
            val = val.v
        
        delta=flag*self.w.立場*w(val)
        if delta>0:
            delta=delta*self.w.立場*-1
        else:
            
            if val==0:
                delta=flag*-1
            elif self.w.立場==0:
                delta=(flag*w(val)*-1)//2
            
            else:
                delta=0
        


        #-32 1 1 
        self.目前怒氣 +=delta
        m(f"我目前的怒氣為［{self.目前怒氣}/{self.忍耐上限}],立場:{self.w.name}", self)
        if isinstance(self.處理的貨物, Output):
            self.處理的貨物.blame(flag, w=self.w) 

class Species():
    
    def __init__(self, num_eyes, id=None):
        if id is None:
            id = random.randint(0, int(10e7))
        
        self.eyes = []
        self.id = id
        
        self.rate = 0
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
        
        
        
        
        s = math.tanh(s/len(self.eyes))
        m(f"輸出{s}", self)
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
        eyes_str = "|".join([repr(n) for n in self.eyes])
        return f'生物{self.id} with eyes:{eyes_str}'

class Env():
    id = 0
    def __init__(self, *shape,id=None):
        if len(shape) != 2:
            raise Exception("目前只開放二維")
        if id ==None:
            Env.id += 1
            self.id = Env.id
        else:
            self.id=id
        dim1 = shape[0]
        dim2 = shape[1]
        self.dim = dim2
        self.species = []
        for n in range(dim2):
            self.species.append(Species(dim1, id=f'{self.id}-{n}'))
        self.species.sort(key=lambda x: x.rate)
    
    def handle(self, val):
        out = []
        if len(self.species[0].eyes) != len(val):
            raise Exception(f'Env Handle 維度不匹配! 輸入元素{val}，但生物有{len(self.species[0].eyes)}')
            
        for n in self.species:
            opt = n.output(val)
            m(f'收集到來自{n}的輸出{opt}', self)
            out.append(opt)
            
        m(f'輸出{out}', self)
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
                m(f'問責{n}', self)
                self.species[n].blame(loss[n])

    def 反應(self):
        for n in self.species:
            m(f'觸發{n}', self)
            n.反應()
    def __str__(self):
        return self.species.__repr__()
target=[0.5,0.2]
if __name__ == '__main__':
    k = Env(2, 30,id='k') 
    g = Env(30, 60,id='g') 
    p = Env(60, 2,id='m') 
    
    

    st=time.time()
    for n in range(60):
        x=k([-1,1])
        x=g(x)
        x=p(x)
        
        flag=loss_fn(x,target)
        
        print([n.v for n in x])
        print(flag)
        p.blame(flag)
        p.反應()
        
        print(analy([k,g,p]),n)
        
    endt=time.time()
    print('用時:',endt-st)
    
        
    
   
    #g=Env(2,2,id='g')
    
    
    
   
    #print(analy([g,k]),'epoch',n)
    
    
    
    
#-32 
