import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(-3, 57); ax.axis("off")
BLUE="#2b5fa8"; LB="#e8f0fb"; OR="#c8641e"; LO="#fdeee0"; GR="#2e7d4f"; LG="#e6f4ea"; GY="#555555"

def box(x,y,w,h,fc,ec,txt,fs=5.4):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.2,rounding_size=1.0",fc=fc,ec=ec,lw=0.9))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color="#111",linespacing=1.35)
def arrow(a,b,c=GY,lw=0.9,ls="-"):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=7,lw=lw,color=c,ls=ls))

ax.text(11,52.5,"Input layer",ha="center",fontsize=6.6,fontweight="bold")
ax.text(11,49.6,"a⁽⁰⁾ = x ∈ ℝ⁴⁵",ha="center",fontsize=6.0)
groups=[("3 scaled numeric\ntenure, MonthlyCharges,\nTotalCharges",39),("1 binary\nSeniorCitizen",28),("41 one-hot columns\n(15 categorical\nvariables)",17)]
for t,y in groups: box(1,y-4.5,20,9,LB,BLUE,t,fs=5.2)
inx=30
iny=[44,38,32,26,20,14,8]
for y in iny: ax.add_patch(Circle((inx,y),1.6,fc="white",ec=BLUE,lw=0.9))
ax.text(inx,3.4,"45 inputs",ha="center",fontsize=5.6,color=GY)
for _,y in groups: arrow((21.8,y),(inx-2.0,y),c=BLUE,lw=0.6)

hx=53
for y in iny:
    ax.add_patch(Circle((hx,y),1.6,fc=LB,ec=BLUE,lw=0.9))
    for yi in iny: ax.plot([inx+1.6,hx-1.6],[yi,y],color="#9bb7dc",lw=0.18,zorder=0)
ax.text(hx,52.5,"Hidden layer",ha="center",fontsize=6.6,fontweight="bold")
ax.text(hx,49.6,"32 ReLU units",ha="center",fontsize=6.0)
ax.text((inx+hx)/2,46.3,"W⁽¹⁾ ∈ ℝ³²ˣ⁴⁵, b⁽¹⁾ ∈ ℝ³²\n1,440 + 32 params",ha="center",va="bottom",fontsize=5.2,color=BLUE)
ax.text(hx,3.4,"32 units",ha="center",fontsize=5.6,color=GY)
ax.text((inx+hx)/2,-0.6,"z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾,   a⁽¹⁾ = ReLU(z⁽¹⁾)",ha="center",fontsize=5.8)

ox=73; oy=28
for y in iny: ax.plot([hx+1.6,ox-2.4],[y,oy],color="#e0a77f",lw=0.18,zorder=0)
ax.add_patch(Circle((ox,oy),2.5,fc=LO,ec=OR,lw=1.1))
ax.text(ox,oy,"σ",ha="center",va="center",fontsize=8,color=OR,fontweight="bold")
ax.text(ox,52.5,"Output layer",ha="center",fontsize=6.6,fontweight="bold")
ax.text(ox,49.6,"1 sigmoid unit",ha="center",fontsize=6.0)
ax.text(ox,41.5,"W⁽²⁾ ∈ ℝ¹ˣ³², b⁽²⁾ ∈ ℝ\n32 + 1 params",ha="center",va="center",fontsize=5.2,color=OR)
ax.text(ox,20.5,"p = σ(z⁽²⁾)\nchurn probability",ha="center",va="center",fontsize=5.8,linespacing=1.35)
ax.text(ox,14.5,"p ≥ 0.5 → churn",ha="center",va="center",fontsize=5.4,color=GY)

arrow((ox+2.9,oy),(83.0,oy),c=OR)
box(83.5,34,15.5,16,LO,OR,"Loss\nBCE + (λ/2)Σ‖W⁽ˡ⁾‖²\nλ = 10⁻⁴\nbatch B = 128",fs=5.0)
box(83.5,9,15.5,19,LG,GR,"Backprop + Adam\nδ⁽ᴸ⁾ = (p − y)/B\nδ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ ReLU′\nlr = 0.003\nβ₁ = 0.9, β₂ = 0.999",fs=4.7)
arrow((91.2,34),(91.2,28.4),c=OR)
arrow((83.2,18),(78.0,18),c=GR,ls="--")
ax.text(80.6,15.9,"update",ha="center",fontsize=4.6,color=GR)
ax.text(91.2,5.8,"Early stopping on\nvalidation loss\n(patience 20)",ha="center",va="center",fontsize=4.9,color=GR,linespacing=1.3)
ax.text(1,-2.4,"Trainable parameters: 1,440 + 32 + 32 + 1 = 1,505.  Init: N(0, 2/n_in) for the hidden layer, N(0, 1/n_in) for the output layer; biases start at 0.",fontsize=4.9,color=GY,ha="left")
plt.savefig("fig_mlp.png",bbox_inches="tight",pad_inches=0.06)
