import numpy as np
import cmath

def get_subsyt_statevectors(sv, target_qubits):
    """
    Get list of statevectors of target qubits where each element of the list is of the form
    [[i], sv, sv_norm] where sv is the normalized statevector of the target qubits when the
    remaining qubits are in the i-th basis state and sv_norm is the total probability of the
    remaining qubits being in the i-th basis state.
    """
    sv=np.asarray(sv)
    n=int(np.log2(len(sv)))
    remaining_qubits=[i for i in range(n) if i not in target_qubits]
    sv_list=[]
    for i in range(2**len(remaining_qubits)):
        sv_el=[]
        active_remaining_qubits=[int(c) for c in str(bin(i))[2:][::-1]]
        index_=sum([on*2**q_num for on,q_num in zip(active_remaining_qubits,remaining_qubits)])
        for j in range(2**len(target_qubits)):
            active_target_qubits=[int(c) for c in str(bin(j))[2:][::-1]]
            index=index_+sum([on*2**q_num for on,q_num in zip(active_target_qubits,target_qubits)])
            sv_el+=[sv[index]]
        sv_el_norm=(sum([np.abs(el)**2 for el in sv_el]))**0.5
        sv_el=np.array(sv_el)
        first_el_phase=cmath.phase(sv_el[0])
        sv_el=sv_el*cmath.exp(-first_el_phase*1j)
        if sv_el_norm:
            sv_el=sv_el/sv_el_norm
        sv_list+=[[[i],sv_el,sv_el_norm**2]]

    return sv_list

def disp_subsyst_statevector(sv,target_qubits,filter_thresh=0.003,remaining_qubit_states=None):
    sv_list=get_subsyt_statevectors(sv, target_qubits)
    n=int(np.log2(len(sv)))
    if remaining_qubit_states==None:
        remaining_qubit_states=list(range(2**(n-len(target_qubits))))
    change=True
    while change:
        change=False
        no_el=len(sv_list)
        for i in range(no_el):
            i=no_el-1-i
            sv=sv_list[i]
            if sv[0][0] not in remaining_qubit_states:
                sv_list.pop(i)
                change=True
                continue
            if sv[2]<filter_thresh:
                sv_list.pop(i)
                change=True
                continue
            for j in range(i):
                if np.allclose(sv_list[j][1],sv[1]) and sv_list[j][0] in remaining_qubit_states:
                    sv_list[j][0]+=sv[0]
                    sv_list[j][0]=list(set(sv_list[j][0]))
                    sv_list[j][2]+=sv[2]
                    sv_list.pop(i)
                    change=True
                    break
    sv_list.sort(key=lambda sv:sv[2])
    for sv in sv_list:
        print("When remaining qubits are in states", ",".join(["|"+(str(bin(ind))[2:]).zfill(n-len(target_qubits))+">" for ind in sv[0]]), " (probability=",sv[2],"),")
        print("the statevector of qubits", ",".join([str(q) for q in target_qubits]), "is")
        print(np.around(sv[1],3))
