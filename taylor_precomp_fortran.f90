real function fact(n)

    integer, intent(in) :: n
    integer :: i
    ! print *, n

    if (n < 0) error stop 'factorial is singular for negative integers'
    fact = 1.0d0
    do i = 2, n
        fact = fact * i
    enddo

    ! print *, fact
end function fact

recursive subroutine add_to_output(power, s, partition_index, expression, output_expression, n, partition, qubits)
    integer, intent(in) :: power
    integer, intent(in) :: s
    integer, intent(in) :: n
    integer, intent(in) :: partition_index
    integer, dimension(1:2**n), intent(in) :: partition
    integer, dimension(1:2**n) :: partition_new
    real, dimension(1:2**n), intent(in) ::  expression
    real, dimension(1:2**n), intent(out) ::  output_expression
    integer, dimension(1:n), intent(in) :: qubits
    integer, dimension(1:n) :: qubits_new
    integer :: i
    integer :: j
    integer :: k

    partition_new=partition

    if ( partition_index==2**n ) then
        partition_new(2**n)=s
        
        qubits_new=qubits
        if ( s /= 0 ) then
            qubits_new=1
        endif
        ! print *, "Partition", partition
        ! print *, "Qubits", qubits
        ! print *, (/((2**i)*qubits(i+1), i=0,n-1)/)
        j=sum((/((2**i)*qubits_new(i+1), i=0,n-1)/))
        ! print *, output_expression
        ! print *, "j: ", j
        output_expression(j+1)=output_expression(j+1)+ &
            product((/(expression(i+1)**partition_new(i+1), i=0,(2**n)-1)/)) &
            *(fact(power)/product((/(fact(partition_new(i)), i=1,(2**n))/)))
        ! print *, "Updated output:", output_expression
        ! qubits=qubits_new

    else
        do i = 0, s
            partition_new(partition_index)=i
            qubits_new=qubits
            if ( i /= 0 ) then
                j=partition_index-1
                k=1
                do while ( j /= 0 )
                    if ( mod(j, 2**k) /= 0 ) then
                        qubits_new(k)=1
                        j=j-mod(j, 2**k)
                    endif
                    k=k+1
                enddo
            endif
            call add_to_output(power, s-i, partition_index+1, expression, output_expression, n, partition_new, qubits_new)
            ! qubits=qubits_new
        end do
    endif
end subroutine add_to_output

subroutine power_expansion(n, expression, output_expression, power)
    implicit none
    integer, intent(in) :: power
    integer, intent(in) :: n
    real, intent(in) :: expression(2**n)
    real, intent(out) :: output_expression(2**n)
    integer :: qubits(n)

    integer :: partition(2**n)

    ! integer :: partitions(2**n:int((fact((2**n)+power-1))/(fact((2**n)-1)*fact(power-((2**n) - 1)))))
    
    output_expression=0.0
    qubits=0

    call add_to_output(power, power, 1, expression, output_expression, n, partition, qubits)


end subroutine power_expansion

! program precomp
!     implicit none
!     real :: output(8)
!     interface
!         subroutine power_expansion(n, expression, output_expression, power)
!             integer, intent(in) :: power
!             integer, intent(in) :: n
!             real, intent(in) :: expression(2**n)
!             real, intent(out) :: output_expression(2**n)
!             integer :: qubits(n)
        
!             integer :: partition(2**n)
!         end subroutine
!     endinterface
!     call power_expansion(3, (/ 0.0,3.0,5.0,0.0,7.0,0.0,0.0,0.0 /), output_expression=output, power=2)
!     print *, output
! end program precomp