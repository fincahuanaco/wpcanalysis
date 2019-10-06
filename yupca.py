import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import ArrowStyle
plt.figure(figsize=(6, 5))
ax=plt.gca()

def aroundcircle(X,rd):
  cx=np.average(X[:,0])
  cy=np.average(X[:,1])
  R=[]
  for r in X:
    #print(r)
    if (r[0]**2+r[1]**2-rd**2)<0:
      R.append(r)
  return np.array(R)

def filterCircle(M,x,y,r):
  R=[]
  for p in M:
    #print(r)
    if ((x-p[0])**2+(y-p[1])**2-r**2)<0:
      R.append(p)
  R=np.array(R)
  print(R.shape)
  cx=np.average(R[:,0])
  cy=np.average(R[:,1])
  return cx,cy,R

def zPCA(data, num_components=None):
  mu = data.mean(axis=0)
  data = data - mu

  evectors, evalues, V = np.linalg.svd(data.T, full_matrices=False)
  projected_data = np.dot(data, evectors)
  sigma = projected_data.std(axis=0).mean()
  print("* ",evectors,sigma)
  return projected_data, evalues, evectors


def drawCircleAxis(cx,cy,T,V,E):
  cx=np.average(T[:,0])+cx
  cy=np.average(T[:,1])+cy
  sigma = T.std(axis=0).mean()
  colors=["red","green"]
  k=0
  for e in E: #each Eigen vector
    start, end = [cx,cy], [cx,cy] + sigma * e * 2
    plt.annotate("", xy=end, xytext=start, arrowprops=dict(facecolor=colors[k], arrowstyle="wedge"))
    k=k+1

def drawCircle(cx,cy,r,color,ls="--",fill=False):
  circle1 = plt.Circle((cx, cy), r, color=color, clip_on=False,fill=fill,lw=2,ls=ls)
  ax.add_artist(circle1)

def drawCircleEvaluate(X,cx,cy,r):
  px,py,P=filterCircle(X,cx,cy,r)
  drawCircle(px,py,0.04,"brown","-",True)
  drawCircle(cx,cy,r,"g")
  R=[]
  for p in P:
    v=p-[px,py]
    R.append(v)
  P=np.array(R)
  CT,CV,CE=zPCA(P,2)
  drawCircleAxis(px,py,CT,CV,CE)

rng = np.random.RandomState(0)
n, d = 2750, 2
X = rng.randn(n, d)  # spherical data
X=aroundcircle(X,1.5)

px=X[:,0]
py=X[:,1]
plt.scatter(px,py,s=5,color="gray")

drawCircleEvaluate(X,1,1,0.5)
drawCircleEvaluate(X,0,0,0.5)

drawCircleEvaluate(X,1.5,0,0.5)
drawCircleEvaluate(X,-1,-1,0.5)
ax.set_aspect('equal', adjustable='datalim')

plt.show()

