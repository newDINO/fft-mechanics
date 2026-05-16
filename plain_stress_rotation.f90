subroutine umat(stress,statev,ddsdde,sse,spd,scd, &
rpl,ddsddt,drplde,drpldt, &
stran,dstran,time,dtime,temp,dtemp,predef,dpred,cmname, &
ndi,nshr,ntens,nstatv,props,nprops,coords,drot,pnewdt, &
celent,dfgrd0,dfgrd1,noel,npt,layer,kspt,jstep,kinc)

    include 'aba_param.inc'

    interface
        subroutine run_with_rotation(ddsdde, stress, stran, dstran, coords, noel, npt) bind(c, name='run_with_rotation')
            real(kind=8) :: ddsdde(3,3), stress(3), stran(3), dstran(3), coords(3)
            integer(kind=4), value :: noel, npt
        end subroutine
    end interface

    character*80 cmname
    dimension stress(ntens),statev(nstatv), &
    ddsdde(ntens,ntens),ddsddt(ntens),drplde(ntens), &
    stran(ntens),dstran(ntens),time(2),predef(1),dpred(1), & 
    props(nprops),coords(3),drot(3,3),dfgrd0(3,3),dfgrd1(3,3), &
    jstep(4)

    call run_with_rotation(ddsdde, stress, stran, dstran, coords, noel, npt)

end subroutine