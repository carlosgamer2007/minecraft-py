"""import re
f = open("world.pyt","r")

i =re.sub("\s+", ",", f.read().strip())
f.close()
f = open("world.pyt","w")
f.write(i)
"""
#FUNCION MAGICA
pos1 = [0,5,0]#X,Y,Z
pos2 = [1,6,0]#X,Y,X
mx = pos2[0] - pos1[0]
my = pos2[1] - pos1[1]
mz = pos2[2] - pos1[2]
for t in range(5):
    x = pos1[0] + mx * t
    y = pos1[1] + my * t
    z = pos1[2] + mz * t
print(x,y,z)


"""

                
                z = int(p)*0.5
                pos1 = [x,y,z]#X,Y,Z
                x+=1
                pos2 = [x,l,z]#X,Y,X
                mx = pos2[0] - pos1[0]
                my = pos2[1] - pos1[1]
                mz = pos2[2] - pos1[2]
                for t in range(4):
                    pa = pos1[0] + mx * t
                    pe = pos1[1] + my * t
                    pi = pos1[2] + mz * t
                    self.add_block((pa,pe,pi),GRASS)

                p = ""
            else:
                print("Formato de archivo no admitido")
            if x ==400:
                z+=1
                x=0
            elif z ==800:
                break
"""