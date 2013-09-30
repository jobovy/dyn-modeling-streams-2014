import sys
import os, os.path
import numpy
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
        pXYZ= bovy_coords.galcenrect_to_XYZ(pvec[0,:],pvec[2,:],pvec[1,:],
                                            Xsun=8.)
        plbd= bovy_coords.XYZ_to_lbd(pXYZ[0],pXYZ[1],pXYZ[2],degree=True)
    #Plot
    bovy_plot.bovy_print(fig_width=8.25,fig_height=3.5)
    if 'ld' in plotfilename:
        lbindx= 2
        ylabel=r'$\mathrm{Distance}\,(\mathrm{kpc})$'
        yrange=[0.,30.]
    else:
        lbindx= 1 
        yrange=[-10.,60.]
        ylabel=r'$\mathrm{Galactic\ latitude}\,(\mathrm{deg})$'
    bovy_plot.bovy_plot(lbd[:,0],lbd[:,lbindx],'k,',
                        xlabel=r'$\mathrm{Galactic\ longitude}\,(\mathrm{deg})$',
                        ylabel=ylabel,
                        xrange=[0.,290.],
                        yrange=yrange)
    if includeorbit:
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
    elif 'arap' in plotfilename:
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
    bovy_plot.bovy_print()
    bovy_plot.bovy_plot(plotx,ploty,'k,',
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

if __name__ == '__main__':
    if 'xz' in sys.argv[1]:
        plot_stream_xz(sys.argv[1])
    if 'lb' in sys.argv[1] or 'ld' in sys.argv[1]:
        plot_stream_lb(sys.argv[1])
    elif 'aa' in sys.argv[1]:
        plot_stream_aa(sys.argv[1])
