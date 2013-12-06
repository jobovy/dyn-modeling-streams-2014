import sys
import os, os.path
import copy
import numpy
from scipy import optimize, special
from galpy.util import bovy_plot, bovy_coords, bovy_conversion
from galpy import potential
from galpy.orbit import Orbit
_STREAMSNAPDIR= '../sim/snaps'
_STREAMSNAPAADIR= '../sim/snaps_aai'
def plot_stream_xz(plotfilename):
    #Read stream
    data= numpy.loadtxt(os.path.join(_STREAMSNAPDIR,'gd1_evol_hitres_01312.dat'),
                        delimiter=',')
    includeorbit= True
    if includeorbit:
        npts= 201
        pot= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
        pts= numpy.linspace(0.,4.,npts)
        #Calculate progenitor orbit around this point
        pox= numpy.median(data[:,1])
        poy= numpy.median(data[:,3])
        poz= numpy.median(data[:,2])
        povx= numpy.median(data[:,4])
        povy= numpy.median(data[:,6])
        povz= numpy.median(data[:,5])
        pR,pphi,pZ= bovy_coords.rect_to_cyl(pox,poy,poz)
        pvR,pvT,pvZ= bovy_coords.rect_to_cyl_vec(povx,povy,povz,pR,
                                                 pphi,pZ,cyl=True)
        ppo= Orbit([pR/8.,pvR/220.,pvT/220.,pZ/8.,pvZ/220.,pphi])
        pno= Orbit([pR/8.,-pvR/220.,-pvT/220.,pZ/8.,-pvZ/220.,pphi])
        ppo.integrate(pts,pot)
        pno.integrate(pts,pot)
        pvec= numpy.zeros((3,npts*2-1))
        pvec[0,:npts-1]= pno.x(pts)[::-1][:-1]
        pvec[1,:npts-1]= pno.z(pts)[::-1][:-1]
        pvec[2,:npts-1]= pno.y(pts)[::-1][:-1]
        pvec[0,npts-1:]= ppo.x(pts)
        pvec[1,npts-1:]= ppo.z(pts)
        pvec[2,npts-1:]= ppo.y(pts)
        pvec*= 8.
    #Plot
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(data[:,1],data[:,2],'k,',
                        xlabel=r'$X\,(\mathrm{kpc})$',
                        ylabel=r'$Z\,(\mathrm{kpc})$',
                        xrange=[0.,16.],
                        yrange=[-0.5,12.])
    if includeorbit:
        bovy_plot.bovy_plot(pox,poz,'o',color='0.5',mec='none',overplot=True,ms=8)
        bovy_plot.bovy_plot(pvec[0,:],pvec[1,:],'k--',overplot=True)
    bovy_plot.bovy_end_print(plotfilename)

def plot_stream_lb(plotfilename):
    #Read stream
    data= numpy.loadtxt(os.path.join(_STREAMSNAPDIR,'gd1_evol_hitres_01312.dat'),
                        delimiter=',')
    #Transform to (l,b)
    XYZ= bovy_coords.galcenrect_to_XYZ(data[:,1],data[:,3],data[:,2],Xsun=8.)
    lbd= bovy_coords.XYZ_to_lbd(XYZ[0],XYZ[1],XYZ[2],degree=True)
    vXYZ= bovy_coords.galcenrect_to_vxvyvz(data[:,4],data[:,6],data[:,5],
                                           vsun=[0.,30.24*8.,0.])
    vlbd= bovy_coords.vxvyvz_to_vrpmllpmbb(vXYZ[0],vXYZ[1],vXYZ[2],
                                           lbd[:,0],lbd[:,1],lbd[:,2],
                                           degree=True)
    includeorbit= True
    if includeorbit:
        npts= 201
        pot= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
        pts= numpy.linspace(0.,4.,npts)
        #Calculate progenitor orbit around this point
        pox= numpy.median(data[:,1])
        poy= numpy.median(data[:,3])
        poz= numpy.median(data[:,2])
        povx= numpy.median(data[:,4])
        povy= numpy.median(data[:,6])
        povz= numpy.median(data[:,5])
        pR,pphi,pZ= bovy_coords.rect_to_cyl(pox,poy,poz)
        pvR,pvT,pvZ= bovy_coords.rect_to_cyl_vec(povx,povy,povz,pR,
                                                 pphi,pZ,cyl=True)
        ppo= Orbit([pR/8.,pvR/220.,pvT/220.,pZ/8.,pvZ/220.,pphi])
        pno= Orbit([pR/8.,-pvR/220.,-pvT/220.,pZ/8.,-pvZ/220.,pphi])
        ppo.integrate(pts,pot)
        pno.integrate(pts,pot)
        pvec= numpy.zeros((6,npts*2-1))
        pvec[0,:npts-1]= pno.x(pts)[::-1][:-1]
        pvec[1,:npts-1]= pno.z(pts)[::-1][:-1]
        pvec[2,:npts-1]= pno.y(pts)[::-1][:-1]
        pvec[0,npts-1:]= ppo.x(pts)
        pvec[1,npts-1:]= ppo.z(pts)
        pvec[2,npts-1:]= ppo.y(pts)
        pvec[3,:npts-1]= -pno.vx(pts)[::-1][:-1]
        pvec[4,:npts-1]= -pno.vz(pts)[::-1][:-1]
        pvec[5,:npts-1]= -pno.vy(pts)[::-1][:-1]
        pvec[3,npts-1:]= ppo.vx(pts)
        pvec[4,npts-1:]= ppo.vz(pts)
        pvec[5,npts-1:]= ppo.vy(pts)
        pvec[:3,:]*= 8.
        pvec[3:,:]*= 220.
        pXYZ= bovy_coords.galcenrect_to_XYZ(pvec[0,:],pvec[2,:],pvec[1,:],
                                            Xsun=8.)
        plbd= bovy_coords.XYZ_to_lbd(pXYZ[0],pXYZ[1],pXYZ[2],degree=True)
        pvXYZ= bovy_coords.galcenrect_to_vxvyvz(pvec[3,:],pvec[5,:],pvec[4,:],
                                                vsun=[0.,30.24*8.,0.])
        pvlbd= bovy_coords.vxvyvz_to_vrpmllpmbb(pvXYZ[0],pvXYZ[1],pvXYZ[2],
                                                plbd[:,0],plbd[:,1],plbd[:,2],
                                                degree=True)
    #Plot
    bovy_plot.bovy_print(fig_width=8.25,fig_height=3.5)
    if 'ld' in plotfilename:
        lbindx= 2
        ylabel=r'$\mathrm{Distance}\,(\mathrm{kpc})$'
        yrange=[0.,30.]
    elif 'lvlos' in plotfilename:
        lbindx= 0
        ylabel=r'$V_\mathrm{los}\,(\mathrm{km\,s}^{-1})$'
        yrange=[-500.,500.]
    else:
        lbindx= 1 
        yrange=[-10.,60.]
        ylabel=r'$\mathrm{Galactic\ latitude}\,(\mathrm{deg})$'
    if 'vlos' in plotfilename:
        bovy_plot.bovy_plot(lbd[:,0],vlbd[:,lbindx],'k,',
                            xlabel=r'$\mathrm{Galactic\ longitude}\,(\mathrm{deg})$',
                            ylabel=ylabel,
                            xrange=[0.,290.],
                            yrange=yrange)
    else:
        bovy_plot.bovy_plot(lbd[:,0],lbd[:,lbindx],'k,',
                            xlabel=r'$\mathrm{Galactic\ longitude}\,(\mathrm{deg})$',
                            ylabel=ylabel,
                            xrange=[0.,290.],
                            yrange=yrange)
    if includeorbit:
        if 'vlos' in plotfilename:
            bovy_plot.bovy_plot(plbd[npts,0],pvlbd[npts,lbindx],
                                'o',color='0.5',mec='none',overplot=True,ms=8)
            bovy_plot.bovy_plot(plbd[:,0],pvlbd[:,lbindx],'k--',overplot=True)
        else:
            bovy_plot.bovy_plot(plbd[npts,0],plbd[npts,lbindx],
                                'o',color='0.5',mec='none',overplot=True,ms=8)
            bovy_plot.bovy_plot(plbd[:,0],plbd[:,lbindx],'k--',overplot=True)
    bovy_plot.bovy_end_print(plotfilename)

def plot_stream_aa(plotfilename):
    #Read stream
    data= numpy.loadtxt(os.path.join(_STREAMSNAPAADIR,
                                     'gd1_evol_hitres_aa_01312.dat'),
                        delimiter=',')
    includeorbit= True
    fmt= 'k,'
    if includeorbit:
        #Read progenitor actions
        progfile= '../sim/gd1_evol_hitres_progaai.dat'
        progaa= numpy.loadtxt(progfile,delimiter=',')
    if 'araz' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        plotx= data[indx,6]
        ploty= data[indx,8]
        plotx= (numpy.pi+(plotx-numpy.median(data[:,6]))) % (2.*numpy.pi)
        ploty= (numpy.pi+(ploty-numpy.median(data[:,8]))) % (2.*numpy.pi)
        xrange=[numpy.pi-1.,numpy.pi+1.]
        yrange=[numpy.pi-1.,numpy.pi+1.]
        xlabel=r'$\theta_R$'
        ylabel=r'$\theta_Z$'
    elif 'arap' in plotfilename and not 'aparaperp' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        plotx= data[indx,6]
        ploty= data[indx,7]
        plotx= (numpy.pi+(plotx-numpy.median(data[:,6]))) % (2.*numpy.pi)
        ploty= (numpy.pi+(ploty-numpy.median(data[:,7]))) % (2.*numpy.pi)
        xrange=[numpy.pi-1.,numpy.pi+1.]
        yrange=[numpy.pi-1.,numpy.pi+1.]
        xlabel=r'$\theta_R$'
        ylabel=r'$\theta_\phi$'
    elif 'oroz' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        plotx= data[indx,3]*bovy_conversion.freq_in_Gyr(220.,8.)
        ploty= data[indx,5]*bovy_conversion.freq_in_Gyr(220.,8.)
        xrange=[15.45,15.95]
        yrange=[11.7,12.05]
        xlabel=r'$\Omega_R\,(\mathrm{Gyr}^{-1})$'
        ylabel=r'$\Omega_Z\,(\mathrm{Gyr}^{-1})$'
    elif 'orop' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        plotx= data[indx,3]*bovy_conversion.freq_in_Gyr(220.,8.)
        ploty= data[indx,4]*bovy_conversion.freq_in_Gyr(220.,8.)
        xrange=[15.45,15.95]
        yrange=[-10.98,-10.65]
        xlabel=r'$\Omega_R\,(\mathrm{Gyr}^{-1})$'
        ylabel=r'$\Omega_\phi\,(\mathrm{Gyr}^{-1})$'
    elif 'jrjz' in plotfilename:       
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        plotx= data[indx,0]*8.
        ploty= data[indx,2]*8.
        xrange=[1.2,1.42]
        yrange=[3.98,4.18]
        xlabel=r'$J_R\,(220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$'
        ylabel=r'$J_Z\,(220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$'
    elif 'jrjp' in plotfilename:       
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        plotx= data[indx,0]*8.
        ploty= data[indx,1]*8.
        xrange=[1.2,1.42]
        yrange=[-14.64,-14.23]
        xlabel=r'$J_R\,(220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$'
        ylabel=r'$L_Z\,(220\,\mathrm{km\,s}^{-1}\,\mathrm{kpc})$'
    elif 'dohist' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        #Frequencies
        Or= data[:,3]
        Op= data[:,4]
        Oz= data[:,5]
        dOr= Or[indx]-numpy.median(Or)
        dOp= Op[indx]-numpy.median(Op)
        dOz= Oz[indx]-numpy.median(Oz)
        dO= numpy.vstack((dOr,dOp,dOz))*bovy_conversion.freq_in_Gyr(220.,8.)
        dO4dir= copy.copy(dO)
        dO4dir[:,dO4dir[:,0] < 0.]*= -1.
        dOdir= numpy.median(dO4dir,axis=1)
        dOdir/= numpy.sqrt(numpy.sum(dOdir**2.))
        dO1d= numpy.dot(dOdir,dO)
        dO1d[dO1d < 0.]*= -1.
        bovy_plot.bovy_print()
        bovy_plot.bovy_hist(dO1d,range=[0.,0.4],bins=61,
                           normed=True,
                           xlabel=r'$|\Delta \mathbf{\Omega}|\,(\mathrm{Gyr}^{-1})$',
                            histtype='step',color='k',zorder=10)
        #Overplot best-fit Gaussian
        xs= numpy.linspace(0.,0.4,1001)
        bovy_plot.bovy_plot(xs,1./numpy.sqrt(2.*numpy.pi)/numpy.std(dO1d)\
                                *numpy.exp(-(xs-numpy.mean(dO1d))**2./2./numpy.var(dO1d)),
                            '--',color='k',overplot=True,lw=2.,zorder=0)
        bestfit= optimize.fmin_powell(gausstimesvalue,
                                      numpy.array([numpy.log(numpy.mean(dO1d)*2.),
                                                   numpy.log(numpy.std(dO1d))]),
                                      args=(dO1d,))
        bovy_plot.bovy_plot(xs,gausstimesvalue(bestfit,xs,nologsum=True),
                            '-',color='0.4',overplot=True,lw=2.,zorder=0)
        bovy_plot.bovy_end_print(plotfilename)
        return None
    elif 'dahist' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        thetar= thetar[indx]
        thetap= data[:,7]
        thetap= (numpy.pi+(thetap-numpy.median(thetap))) % (2.*numpy.pi)
        thetap= thetap[indx]
        thetaz= data[:,8]
        thetaz= (numpy.pi+(thetaz-numpy.median(thetaz))) % (2.*numpy.pi)
        thetaz= thetaz[indx]
        #center around 0 (instead of pi)
        thetar-= numpy.pi
        thetap-= numpy.pi
        thetaz-= numpy.pi
        #Frequencies
        Or= data[:,3]
        Op= data[:,4]
        Oz= data[:,5]
        dOr= Or[indx]-numpy.median(Or)
        dOp= Op[indx]-numpy.median(Op)
        dOz= Oz[indx]-numpy.median(Oz)
        dO= numpy.vstack((dOr,dOp,dOz))*bovy_conversion.freq_in_Gyr(220.,8.)
        #Direction in which the stream spreads
        dO4dir= copy.copy(dO)
        dO4dir[:,dO4dir[:,0] < 0.]*= -1.
        dOdir= numpy.median(dO4dir,axis=1)
        dOdir/= numpy.sqrt(numpy.sum(dOdir**2.))
        #Gram-Schmidt to get the perpendicular directions
        v2= numpy.array([1.,0.,0.])
        v3= numpy.array([0.,1.,0.])
        u2= v2-numpy.sum(dOdir*v2)*dOdir
        u2/= numpy.sqrt(numpy.sum(u2**2.))
        u3= v3-numpy.sum(dOdir*v3)*dOdir-numpy.sum(u2*v3)*u2
        #Times
        dangle= numpy.vstack((thetar,thetap,thetaz))
        dts= numpy.sum(dO*dangle,axis=0)/numpy.sum(dO**2.,axis=0)
        #Rewind angles
        dangle-= dO*dts
        newdangle= numpy.empty_like(dangle)
        newdangle[0,:]= numpy.dot(dOdir,dangle)
        newdangle[1,:]= numpy.dot(u2,dangle)
        newdangle[2,:]= numpy.dot(u3,dangle)
        bovy_plot.bovy_print()
        xmin= -0.015
        bovy_plot.bovy_hist(newdangle[2,:].flatten(),range=[xmin,-xmin],bins=61,
                            xlabel=r'$|\Delta \mathbf{\theta}|$',
                            normed=True,lw=2.,
                            color='k',zorder=10,
                            histtype='step')
        #Overplot best-fit Gaussian
        xs= numpy.linspace(xmin,-xmin,1001)
        bovy_plot.bovy_plot(xs,1./numpy.sqrt(2.*numpy.pi)/numpy.std(newdangle[1:,:])\
                                *numpy.exp(-(xs-numpy.mean(newdangle[1:,:]))**2./2./numpy.var(newdangle[1:,:])),
                            '--',color='k',overplot=True,lw=2.,zorder=0)
        print "along", numpy.mean(newdangle[0,:]), numpy.std(newdangle[0,:])
        print "perpendicular 1", numpy.mean(newdangle[1,:]), numpy.std(newdangle[1,:])
        print "perpendicular 2", numpy.mean(newdangle[2,:]), numpy.std(newdangle[2,:])
        bovy_plot.bovy_end_print(plotfilename)
        return None
    elif 'aparopar' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        thetar= thetar[indx]
        thetap= data[:,7]
        thetap= (numpy.pi+(thetap-numpy.median(thetap))) % (2.*numpy.pi)
        thetap= thetap[indx]
        thetaz= data[:,8]
        thetaz= (numpy.pi+(thetaz-numpy.median(thetaz))) % (2.*numpy.pi)
        thetaz= thetaz[indx]
        #center around 0 (instead of pi)
        thetar-= numpy.pi
        thetap-= numpy.pi
        thetaz-= numpy.pi
        #Frequencies
        Or= data[:,3]
        Op= data[:,4]
        Oz= data[:,5]
        dOr= Or[indx]-numpy.median(Or)
        dOp= Op[indx]-numpy.median(Op)
        dOz= Oz[indx]-numpy.median(Oz)
        dO= numpy.vstack((dOr,dOp,dOz))*bovy_conversion.freq_in_Gyr(220.,8.)
        #Direction in which the stream spreads
        dO4dir= copy.copy(dO)
        dO4dir[:,dO4dir[:,0] < 0.]*= -1.
        dOdir= numpy.median(dO4dir,axis=1)
        dOdir/= numpy.sqrt(numpy.sum(dOdir**2.))
        #Times
        dangle= numpy.vstack((thetar,thetap,thetaz))
        plotx= numpy.fabs(numpy.dot(dangle.T,dOdir))
        ploty= numpy.fabs(numpy.dot(dO.T,dOdir))
        xrange=[0.,1.3]
        yrange=[0.1,0.3]
        xlabel= r'$\Large|\Delta \mathbf{\theta}_\parallel\Large|$'
        ylabel= r'$\Large|\Delta \mathbf{\Omega}_\parallel\Large|\,(\mathrm{Gyr}^{-1})$'
        fmt= 'k.'
    elif 'aparoperp' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        thetar= thetar[indx]
        thetap= data[:,7]
        thetap= (numpy.pi+(thetap-numpy.median(thetap))) % (2.*numpy.pi)
        thetap= thetap[indx]
        thetaz= data[:,8]
        thetaz= (numpy.pi+(thetaz-numpy.median(thetaz))) % (2.*numpy.pi)
        thetaz= thetaz[indx]
        #center around 0 (instead of pi)
        thetar-= numpy.pi
        thetap-= numpy.pi
        thetaz-= numpy.pi
        #Frequencies
        Or= data[:,3]
        Op= data[:,4]
        Oz= data[:,5]
        dOr= Or[indx]-numpy.median(Or)
        dOp= Op[indx]-numpy.median(Op)
        dOz= Oz[indx]-numpy.median(Oz)
        dO= numpy.vstack((dOr,dOp,dOz))*bovy_conversion.freq_in_Gyr(220.,8.)
        #Direction in which the stream spreads
        dO4dir= copy.copy(dO)
        dO4dir[:,dO4dir[:,0] < 0.]*= -1.
        dOdir= numpy.median(dO4dir,axis=1)
        dOdir/= numpy.sqrt(numpy.sum(dOdir**2.))
        #Times
        dangle= numpy.vstack((thetar,thetap,thetaz))
        plotx= numpy.fabs(numpy.dot(dangle.T,dOdir))
        ploty= numpy.sqrt(numpy.sum(dO**2.,axis=0)\
                              -(numpy.dot(dO.T,dOdir))**2.)
        xrange=[0.,1.3]
        yrange=[0.,0.005]
        xlabel= r'$\Large|\Delta \mathbf{\theta}_\parallel\Large|$'
        ylabel= r'$\Large|\Delta \mathbf{\Omega}_\perp\Large|\,(\mathrm{Gyr}^{-1})$'
        fmt= 'k.'
    elif 'aparaperp' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        thetar= thetar[indx]
        thetap= data[:,7]
        thetap= (numpy.pi+(thetap-numpy.median(thetap))) % (2.*numpy.pi)
        thetap= thetap[indx]
        thetaz= data[:,8]
        thetaz= (numpy.pi+(thetaz-numpy.median(thetaz))) % (2.*numpy.pi)
        thetaz= thetaz[indx]
        #center around 0 (instead of pi)
        thetar-= numpy.pi
        thetap-= numpy.pi
        thetaz-= numpy.pi
        #Frequencies
        Or= data[:,3]
        Op= data[:,4]
        Oz= data[:,5]
        dOr= Or[indx]-numpy.median(Or)
        dOp= Op[indx]-numpy.median(Op)
        dOz= Oz[indx]-numpy.median(Oz)
        dO= numpy.vstack((dOr,dOp,dOz))*bovy_conversion.freq_in_Gyr(220.,8.)
        #Direction in which the stream spreads
        dO4dir= copy.copy(dO)
        dO4dir[:,dO4dir[:,0] < 0.]*= -1.
        dOdir= numpy.median(dO4dir,axis=1)
        dOdir/= numpy.sqrt(numpy.sum(dOdir**2.))
        #Times
        dangle= numpy.vstack((thetar,thetap,thetaz))
        plotx= numpy.fabs(numpy.dot(dangle.T,dOdir))
        ploty= numpy.sqrt(numpy.sum(dangle**2.,axis=0)\
                              -(numpy.dot(dangle.T,dOdir))**2.)
        xrange=[0.,1.3]
        yrange=[0.,0.03]
        xlabel= r'$\Large|\Delta \mathbf{\theta}_\parallel\Large|$'
        ylabel= r'$\Large|\Delta \mathbf{\theta}_\perp\Large|$'
        fmt= 'k.'
    elif 'apartime' in plotfilename:
        thetar= data[:,6]
        thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
        indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
        thetar= thetar[indx]
        thetap= data[:,7]
        thetap= (numpy.pi+(thetap-numpy.median(thetap))) % (2.*numpy.pi)
        thetap= thetap[indx]
        thetaz= data[:,8]
        thetaz= (numpy.pi+(thetaz-numpy.median(thetaz))) % (2.*numpy.pi)
        thetaz= thetaz[indx]
        #center around 0 (instead of pi)
        thetar-= numpy.pi
        thetap-= numpy.pi
        thetaz-= numpy.pi
        #Frequencies
        Or= data[:,3]
        Op= data[:,4]
        Oz= data[:,5]
        dOr= Or[indx]-numpy.median(Or)
        dOp= Op[indx]-numpy.median(Op)
        dOz= Oz[indx]-numpy.median(Oz)
        dO= numpy.vstack((dOr,dOp,dOz))*bovy_conversion.freq_in_Gyr(220.,8.)
        #Direction in which the stream spreads
        dO4dir= copy.copy(dO)
        dO4dir[:,dO4dir[:,0] < 0.]*= -1.
        dOdir= numpy.median(dO4dir,axis=1)
        dOdir/= numpy.sqrt(numpy.sum(dOdir**2.))
        #Times
        dangle= numpy.vstack((thetar,thetap,thetaz))
        dts= numpy.sum(dO*dangle,axis=0)/numpy.sum(dO**2.,axis=0)
        plotx= numpy.fabs(numpy.dot(dangle.T,dOdir))
        ploty= dts
        xrange=[0.,1.3]
        yrange=[0.,5.]
        xlabel= r'$\Large|\Delta \mathbf{\theta}_\parallel\Large|$'
        ylabel= r'$\Delta t\,(\mathrm{Gyr})$'
        fmt= 'k.'
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(plotx,ploty,fmt,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        xrange=xrange,
                        yrange=yrange)
    if includeorbit and 'araz' in plotfilename:
        #plot frequency line
        xs= numpy.array(xrange)
        ys= (xs-numpy.pi)*progaa[-1,5]/progaa[-1,3]+numpy.pi
        bovy_plot.bovy_plot(xs,ys,'k--',overplot=True,
                            zorder=0)
    elif includeorbit and 'arap' in plotfilename:
        #plot frequency line
        xs= numpy.array(xrange)
        ys= (xs-numpy.pi)*progaa[-1,4]/progaa[-1,3]+numpy.pi
        bovy_plot.bovy_plot(xs,ys,'k--',overplot=True,
                            zorder=0)
    elif includeorbit and 'oroz' in plotfilename:
        bovy_plot.bovy_plot(progaa[-1,3]*bovy_conversion.freq_in_Gyr(220.,8.),
                            progaa[-1,5]*bovy_conversion.freq_in_Gyr(220.,8.),
                            'o',overplot=True,color='0.5',
                            mec='none',ms=8.,
                            zorder=0)
    elif includeorbit and 'orop' in plotfilename:
        bovy_plot.bovy_plot(progaa[-1,3]*bovy_conversion.freq_in_Gyr(220.,8.),
                            progaa[-1,4]*bovy_conversion.freq_in_Gyr(220.,8.),
                            'o',overplot=True,color='0.5',
                            mec='none',ms=8.,
                            zorder=0)
    elif includeorbit and 'jrjz' in plotfilename:
        bovy_plot.bovy_plot(progaa[-1,0]*8.,
                            progaa[-1,2]*8.,
                            'o',overplot=True,color='0.5',
                            mec='none',ms=8.,
                            zorder=0)
    elif includeorbit and 'jrjp' in plotfilename:
        bovy_plot.bovy_plot(progaa[-1,0]*8.,
                            progaa[-1,1]*8.,
                            'o',overplot=True,color='0.5',
                            mec='none',ms=8.,
                            zorder=0)
    bovy_plot.bovy_end_print(plotfilename)

def plot_stream_times(plotfilename):
    #Read stream
    data= numpy.loadtxt(os.path.join(_STREAMSNAPAADIR,
                                     'gd1_evol_hitres_aa_01312.dat'),
                        delimiter=',')
    #Calculate times at which stars were stripped, angles
    thetar= data[:,6]
    thetar= (numpy.pi+(thetar-numpy.median(thetar))) % (2.*numpy.pi)
    indx= numpy.fabs(thetar-numpy.pi) > (5.*numpy.median(numpy.fabs(thetar-numpy.median(thetar))))
    thetar= thetar[indx]
    thetap= data[:,7]
    thetap= (numpy.pi+(thetap-numpy.median(thetap))) % (2.*numpy.pi)
    thetap= thetap[indx]
    thetaz= data[:,8]
    thetaz= (numpy.pi+(thetaz-numpy.median(thetaz))) % (2.*numpy.pi)
    thetaz= thetaz[indx]
    #center around 0 (instead of pi)
    thetar-= numpy.pi
    thetap-= numpy.pi
    thetaz-= numpy.pi
    #Frequencies
    Or= data[:,3]
    Op= data[:,4]
    Oz= data[:,5]
    dOr= Or[indx]-numpy.median(Or)
    dOp= Op[indx]-numpy.median(Op)
    dOz= Oz[indx]-numpy.median(Oz)
    #Times
    dangle= numpy.vstack((thetar,thetap,thetaz))
    dO= numpy.vstack((dOr,dOp,dOz))*bovy_conversion.freq_in_Gyr(220.,8.)
    dts= numpy.sum(dO*dangle,axis=0)/numpy.sum(dO**2.,axis=0)
    if 'hist' in plotfilename:
        bovy_plot.bovy_print()
        bovy_plot.bovy_hist(dts,range=[0.,6.],bins=13,
                            xlabel=r'$\Delta t\ (\mathrm{Gyr})$',
                            histtype='step',color='k',normed=True,lw=2.)
        bovy_plot.bovy_hist(dts,range=[0.,6.],bins=61,normed=True,
                            overplot=True,ls='dotted',histtype='step',
                            color='0.5',lw=2.)
        includeorbit= False #bc pericenter passage doesn't seem to work
        if includeorbit:
            npts= 10001
            pot= potential.LogarithmicHaloPotential(normalize=1.,q=0.9)
            ts= numpy.linspace(0.,5./bovy_conversion.time_in_Gyr(220.,8.),
                               npts)
            #Calculate progenitor orbit around this point
            pox= numpy.median(data[:,1])
            poy= numpy.median(data[:,3])
            poz= numpy.median(data[:,2])
            povx= numpy.median(data[:,4])
            povy= numpy.median(data[:,6])
            povz= numpy.median(data[:,5])
            pR,pphi,pZ= bovy_coords.rect_to_cyl(pox,poy,poz)
            pvR,pvT,pvZ= bovy_coords.rect_to_cyl_vec(povx,povy,povz,pR,
                                                     pphi,pZ,cyl=True)
            o= Orbit([pR/8.,-pvR/220.,-pvT/220.,pZ/8.,-pvZ/220.,pphi])
            o.integrate(ts,pot)
            rs= numpy.sqrt(o.R(ts)**2.+o.z(ts)**2.)
            #Find local minima
            periIndx= numpy.r_[True, rs[1:] < rs[:-1]] & numpy.r_[rs[:-1] < rs[1:], True]
            for ii in range(numpy.sum(periIndx)):
                bovy_plot.bovy_plot([ts[periIndx][ii]*bovy_conversion.time_in_Gyr(220.,8.),
                                     ts[periIndx][ii]*bovy_conversion.time_in_Gyr(220.,8.)],
                                    [0.,1.],overplot=True,ls='--',lw=1.,
                                    color='0.5')
    elif 'dO' in plotfilename:
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(dts,
                            numpy.sqrt(numpy.sum(dO**2.,axis=0)),
                            'k.',
                            xrange=[0.,6.],
                            yrange=[0.1,0.35],
                            xlabel=r'$\Delta t\ (\mathrm{Gyr})$',
                            ylabel=r'$|\Delta \mathbf{\Omega}|\ (\mathrm{Gyr}^{-1})$')
    elif 'da' in plotfilename:
        bovy_plot.bovy_print()
        bovy_plot.bovy_plot(dts,
                            numpy.sqrt(numpy.sum(dangle**2.,axis=0)),
                            'k.',
                            xrange=[0.,6.],
                            yrange=[0.,1.5],
                            xlabel=r'$\Delta t\ (\mathrm{Gyr})$',
                            ylabel=r'$|\Delta \boldsymbol\theta|$')
    bovy_plot.bovy_end_print(plotfilename)
    return None

def gausstimesvalue(params,vals,nologsum=False):
    tmean= numpy.exp(params[0])
    tsig= numpy.exp(params[1])
    norm= tsig**2.*numpy.exp(-tmean**2./2./tsig**2.)+tsig*numpy.sqrt(numpy.pi/2.)*tmean*(1.+special.erf(tmean/numpy.sqrt(2.)/tsig))
    if nologsum:
        return numpy.fabs(vals)/norm*numpy.exp(-(vals-tmean)**2./2./tsig**2.)
    else:
        return -numpy.sum(numpy.log(numpy.fabs(vals)/norm*numpy.exp(-(vals-tmean)**2./2./tsig**2.)))

if __name__ == '__main__':
    if 'xz' in sys.argv[1]:
        plot_stream_xz(sys.argv[1])
    if 'lb' in sys.argv[1] or 'ld' in sys.argv[1] or 'lvlos' in sys.argv[1]:
        plot_stream_lb(sys.argv[1])
    elif 'times' in sys.argv[1]:
        plot_stream_times(sys.argv[1])
    elif 'aa' in sys.argv[1]:
        plot_stream_aa(sys.argv[1])
