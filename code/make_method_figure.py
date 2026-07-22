import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT="/sessions/keen-loving-cannon/mnt/ROMBUN_HAKASE_PhD"

def box(ax,x,y,w,h,label,num,fs=9.5,dashed=False,inside=False):
    s=dict(boxstyle="round,pad=0.35,rounding_size=2",linewidth=1.3,edgecolor="black",facecolor="white")
    if dashed:s["linestyle"]=(0,(5,3))
    ax.add_patch(FancyBboxPatch((x,y),w,h,**s))
    ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs)
    if num:
        if inside: ax.text(x+w-0.8,y+h-0.8,num,ha="right",va="top",fontsize=8.5,fontweight="bold")
        else: ax.text(x+w+0.6,y+h+0.4,num,ha="right",va="bottom",fontsize=8.5,fontweight="bold")
def diamond(ax,cx,cy,w,h,label,num,fs=9):
    pts=[(cx,cy+h/2),(cx+w/2,cy),(cx,cy-h/2),(cx-w/2,cy)]
    ax.add_patch(Polygon(pts,closed=True,linewidth=1.3,edgecolor="black",facecolor="white"))
    ax.text(cx,cy,label,ha="center",va="center",fontsize=fs)
    if num:ax.text(cx+w/2+0.5,cy+h/2-0.5,num,ha="left",va="center",fontsize=8.5,fontweight="bold")
def arr(ax,x1,y1,x2,y2,head="-|>"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=head,mutation_scale=12,
                 linewidth=1.2,color="black",shrinkA=1,shrinkB=1))
def lin(ax,x1,y1,x2,y2):
    ax.plot([x1,x2],[y1,y2],color="black",linewidth=1.2)

# ================= FIG 1 =================
f1,ax=plt.subplots(figsize=(8.6,6.8)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
box(ax,2,4,96,92,"","100",dashed=True,inside=True)
ax.text(5,93.5,"Safety-supervision system",fontsize=9,style="italic")
box(ax,6,78,27,9,"Vehicle & environmental\nstate inputs","105")
box(ax,6,64,27,8,"Operator target\nintervention rate  $\\tau$","")
box(ax,40,76,23,11,"Risk computation\nmodule  R(x)","110")
box(ax,5,44,27,10,"Online recalibration\nmodule - boundary B(t)","120",fs=9)
box(ax,37,44,26,10,"Change-point\ndetector (CUSUM)","140",fs=9)
box(ax,68,44,27,10,"Sliding-window\nfinite-sample\ncertificate (DKW)","150",fs=8.5)
box(ax,27,26.5,46,11,"Safety-clamp module\nslow estimate U(t);  B_eff = min( B(t), U(t)+m )","130",fs=9)
box(ax,27,11.5,46,9,"Supervisory constraint module\nintervene if  R(x$_t$) > B_eff","160",fs=9)
box(ax,77,11.5,17,9,"Vehicle\nactuation","170")
# arrows
arr(ax,33,82.5,40,82.5)                 # inputs -> risk
lin(ax,19,64,19,60); arr(ax,19,60,18.5,54)   # tau -> 120
lin(ax,51.5,76,51.5,58); lin(ax,18.5,58,81.5,58)  # R(x) distribution bus
ax.text(53,60,"R(x$_t$)",fontsize=8,style="italic")
arr(ax,18.5,58,18.5,54); arr(ax,50,58,50,54); arr(ax,81.5,58,81.5,54)
arr(ax,18.5,44,42,37.5)                 # 120 -> clamp (B(t))
lin(ax,81.5,44,81.5,33); arr(ax,81.5,33,71.5,33)   # 150 -> clamp (finite-sample band; ends inside box)
ax.text(78,34.2,"band",fontsize=7,style="italic",ha="center")
arr(ax,37,49,32,49)                     # CUSUM -> 120 (refit trigger)
ax.text(34.5,50.4,"refit",fontsize=7,style="italic",ha="center")
arr(ax,50,26.5,50,20.5)                 # clamp -> supervisory
ax.text(51.5,23.3,"B_eff",fontsize=8,style="italic")
arr(ax,73,16,77,16)                     # supervisory -> actuation
ax.text(50,1.2,"FIG. 1",ha="center",fontsize=12,fontweight="bold")

# ================= FIG 2 =================
f2,ax=plt.subplots(figsize=(7.4,9.8)); ax.set_xlim(0,100); ax.set_ylim(-6,100); ax.axis("off")
cx=44
box(ax,cx-22,92,44,6,"At each time instant  t","200",fs=9)
box(ax,cx-26,82,52,6.5,"Compute risk value  R(x$_t$)","202",fs=9)
box(ax,cx-30,70.5,60,7.5,"Compute slow estimate U(t);\nB_eff = min( B(t), U(t)+m )","203",fs=8.8)
diamond(ax,cx,60,40,11,"R(x$_t$) > B_eff ?","204",fs=8.6)
box(ax,cx+26,57.5,26,6,"Engage\nintervention","206",fs=8.6)
box(ax,cx-32,46,64,7,"Realized-rate error\n e$_t$ = 1[R>B_eff] $-$ target rate","208",fs=8.6)
box(ax,cx-32,35,64,7,"Estimate drift; set step size  \u03b3  (optional)","210",fs=8.6)
box(ax,cx-32,24,64,7,"Update boundary\n B(t+1) = clip( B(t) + $\\gamma\\,$e$_t$ )","212",fs=8.6)
diamond(ax,cx,14.5,44,11,"CUSUM  S$_t$ > thr ?","214",fs=8.4)
box(ax,cx+28,11.5,28,7,"Batch recompute B\nover window; reset S","216",fs=8)
box(ax,cx-34,1,68,6.5,"Re-establish per-window finite-sample (DKW) certificate","218",fs=8.4)
arr(ax,cx,92,cx,88.5)
arr(ax,cx,82,cx,78)
arr(ax,cx,70.5,cx,65.5)
arr(ax,cx+20,60,cx+26,60.5); ax.text(cx+22,61.5,"Y",fontsize=8)
arr(ax,cx,54.5,cx,53); ax.text(cx+1.6,53.9,"N",fontsize=8)
lin(ax,cx+39,57.5,cx+39,49.5); arr(ax,cx+39,49.5,cx+32,49.5)
arr(ax,cx,46,cx,42)
arr(ax,cx,35,cx,31)
arr(ax,cx,24,cx,20)
arr(ax,cx+22,14.5,cx+28,15); ax.text(cx+23,16.1,"Y",fontsize=8)
arr(ax,cx,9,cx,7.5); ax.text(cx+1.6,8.3,"N",fontsize=8)
lin(ax,cx+42,11.5,cx+42,4.25); arr(ax,cx+42,4.25,cx+34,4.25)
lin(ax,cx-34,4.25,3,4.25); lin(ax,3,4.25,3,95); arr(ax,3,95,cx-22,95)
ax.text(2.0,50,"next instant",rotation=90,fontsize=8,style="italic",va="center")
ax.text(50,-4,"FIG. 2",ha="center",fontsize=12,fontweight="bold")

# ================= FIG 3 =================
f3,ax=plt.subplots(figsize=(8.0,5.2))
t=np.linspace(0,10,400); shift=5.0; tau=0.10
fixed=np.where(t<shift,tau+0.004*np.sin(t*3),tau+0.09*(1-np.exp(-(t-shift)/1.6)))
online=tau+0.006*np.sin(t*4)*np.exp(-abs(t-shift)/6)
ax.plot(t,online,color="black",linewidth=1.8,label="online-recalibrated boundary (330)")
ax.plot(t,fixed,color="black",linewidth=1.6,linestyle=(0,(6,4)),label="fixed boundary (320)")
ax.axhline(tau,color="black",linewidth=1.0,linestyle=(0,(1,2)))
ax.axvline(shift,color="black",linewidth=1.0,linestyle=(0,(4,3)))
ax.text(shift-0.15,0.135,"distribution shift onset (340)",rotation=90,ha="right",va="center",fontsize=8.5)
ax.text(0.4,tau+0.006,"target rate $\\tau$ (310)",ha="left",va="bottom",fontsize=8.5)
ax.text(9.7,fixed[-1]+0.003,"320",fontsize=9,fontweight="bold",va="bottom")
ax.text(9.7,online[-1]-0.004,"330",fontsize=9,fontweight="bold",va="top")
ax.set_xlabel("Time (302)",fontsize=10); ax.set_ylabel("Realized intervention rate (304)",fontsize=10)
ax.set_xticks([]); ax.set_yticks([tau]); ax.set_yticklabels(["$\\tau$"])
ax.set_ylim(0.06,0.225); ax.set_xlim(0,10)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.legend(loc="upper left",bbox_to_anchor=(0.02,0.99),fontsize=8.5,frameon=False)
ax.text(5,0.045,"FIG. 3",ha="center",fontsize=12,fontweight="bold",transform=ax.transData)
f3.subplots_adjust(bottom=0.16)

for f,n in [(f1,"fig1"),(f2,"fig2"),(f3,"fig3")]:
    f.savefig(f"{OUT}/_patent_{n}.png",dpi=200,bbox_inches="tight")
with PdfPages(f"{OUT}/Clamp_Patent_Figures_v4.pdf") as pdf:
    for f in (f1,f2,f3): pdf.savefig(f,bbox_inches="tight")
print("figures written")
