
tiny_manna_tasks:     file format elf64-x86-64
tiny_manna_tasks
architecture: i386:x86-64, flags 0x00000150:
HAS_SYMS, DYNAMIC, D_PAGED
start address 0x0000000000002730

Program Header:
    PHDR off    0x0000000000000040 vaddr 0x0000000000000040 paddr 0x0000000000000040 align 2**3
         filesz 0x0000000000000268 memsz 0x0000000000000268 flags r--
  INTERP off    0x00000000000002a8 vaddr 0x00000000000002a8 paddr 0x00000000000002a8 align 2**0
         filesz 0x000000000000001c memsz 0x000000000000001c flags r--
    LOAD off    0x0000000000000000 vaddr 0x0000000000000000 paddr 0x0000000000000000 align 2**12
         filesz 0x0000000000000f38 memsz 0x0000000000000f38 flags r--
    LOAD off    0x0000000000001000 vaddr 0x0000000000001000 paddr 0x0000000000001000 align 2**12
         filesz 0x000000000000217d memsz 0x000000000000217d flags r-x
    LOAD off    0x0000000000004000 vaddr 0x0000000000004000 paddr 0x0000000000004000 align 2**12
         filesz 0x00000000000004c2 memsz 0x00000000000004c2 flags r--
    LOAD off    0x0000000000004d80 vaddr 0x0000000000005d80 paddr 0x0000000000005d80 align 2**12
         filesz 0x0000000000000370 memsz 0x0000000000003668 flags rw-
 DYNAMIC off    0x0000000000004da0 vaddr 0x0000000000005da0 paddr 0x0000000000005da0 align 2**3
         filesz 0x0000000000000230 memsz 0x0000000000000230 flags rw-
    NOTE off    0x00000000000002c4 vaddr 0x00000000000002c4 paddr 0x00000000000002c4 align 2**2
         filesz 0x0000000000000044 memsz 0x0000000000000044 flags r--
EH_FRAME off    0x00000000000041c0 vaddr 0x00000000000041c0 paddr 0x00000000000041c0 align 2**2
         filesz 0x000000000000006c memsz 0x000000000000006c flags r--
   STACK off    0x0000000000000000 vaddr 0x0000000000000000 paddr 0x0000000000000000 align 2**4
         filesz 0x0000000000000000 memsz 0x0000000000000000 flags rw-
   RELRO off    0x0000000000004d80 vaddr 0x0000000000005d80 paddr 0x0000000000005d80 align 2**0
         filesz 0x0000000000000280 memsz 0x0000000000000280 flags r--

Dynamic Section:
  NEEDED               libstdc++.so.6
  NEEDED               libm.so.6
  NEEDED               libgomp.so.1
  NEEDED               libgcc_s.so.1
  NEEDED               libpthread.so.0
  NEEDED               libc.so.6
  INIT                 0x0000000000001000
  FINI                 0x0000000000003174
  INIT_ARRAY           0x0000000000005d80
  INIT_ARRAYSZ         0x0000000000000018
  FINI_ARRAY           0x0000000000005d98
  FINI_ARRAYSZ         0x0000000000000008
  GNU_HASH             0x0000000000000308
  STRTAB               0x0000000000000668
  SYMTAB               0x0000000000000338
  STRSZ                0x00000000000003fa
  SYMENT               0x0000000000000018
  DEBUG                0x0000000000000000
  PLTGOT               0x0000000000006000
  PLTRELSZ             0x0000000000000240
  PLTREL               0x0000000000000007
  JMPREL               0x0000000000000cf8
  RELA                 0x0000000000000ba8
  RELASZ               0x0000000000000150
  RELAENT              0x0000000000000018
  FLAGS_1              0x0000000008000000
  VERNEED              0x0000000000000aa8
  VERNEEDNUM           0x0000000000000004
  VERSYM               0x0000000000000a62
  RELACOUNT            0x0000000000000005

Version References:
  required from libgcc_s.so.1:
    0x0b792650 0x00 13 GCC_3.0
  required from libc.so.6:
    0x06969196 0x00 06 GLIBC_2.16
    0x09691a75 0x00 04 GLIBC_2.2.5
  required from libstdc++.so.6:
    0x0297f871 0x00 12 GLIBCXX_3.4.21
    0x056bafd3 0x00 11 CXXABI_1.3
    0x02297f89 0x00 08 GLIBCXX_3.4.9
    0x0297f868 0x00 07 GLIBCXX_3.4.18
    0x08922974 0x00 03 GLIBCXX_3.4
  required from libgomp.so.1:
    0x04262440 0x00 10 OMP_1.0
    0x042623d0 0x00 09 GOMP_4.0
    0x042621d0 0x00 05 GOMP_2.0
    0x042628d0 0x00 02 GOMP_1.0

Sections:
Idx Name          Size      VMA               LMA               File off  Algn
  0 .interp       0000001c  00000000000002a8  00000000000002a8  000002a8  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  1 .note.ABI-tag 00000020  00000000000002c4  00000000000002c4  000002c4  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  2 .note.gnu.build-id 00000024  00000000000002e4  00000000000002e4  000002e4  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  3 .gnu.hash     00000030  0000000000000308  0000000000000308  00000308  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  4 .dynsym       00000330  0000000000000338  0000000000000338  00000338  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  5 .dynstr       000003fa  0000000000000668  0000000000000668  00000668  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  6 .gnu.version  00000044  0000000000000a62  0000000000000a62  00000a62  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  7 .gnu.version_r 00000100  0000000000000aa8  0000000000000aa8  00000aa8  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  8 .rela.dyn     00000150  0000000000000ba8  0000000000000ba8  00000ba8  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  9 .rela.plt     00000240  0000000000000cf8  0000000000000cf8  00000cf8  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 10 .init         00000017  0000000000001000  0000000000001000  00001000  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 11 .plt          00000190  0000000000001020  0000000000001020  00001020  2**4
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 12 .plt.got      00000008  00000000000011b0  00000000000011b0  000011b0  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 13 .text         00001fb2  00000000000011c0  00000000000011c0  000011c0  2**4
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 14 .fini         00000009  0000000000003174  0000000000003174  00003174  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 15 .rodata       000001c0  0000000000004000  0000000000004000  00004000  2**5
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 16 .eh_frame_hdr 0000006c  00000000000041c0  00000000000041c0  000041c0  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 17 .eh_frame     00000268  0000000000004230  0000000000004230  00004230  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 18 .gcc_except_table 0000002a  0000000000004498  0000000000004498  00004498  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 19 .init_array   00000018  0000000000005d80  0000000000005d80  00004d80  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 20 .fini_array   00000008  0000000000005d98  0000000000005d98  00004d98  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 21 .dynamic      00000230  0000000000005da0  0000000000005da0  00004da0  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 22 .got          00000030  0000000000005fd0  0000000000005fd0  00004fd0  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 23 .got.plt      000000d8  0000000000006000  0000000000006000  00005000  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 24 .data         00000018  00000000000060d8  00000000000060d8  000050d8  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 25 .bss          000032e8  0000000000006100  0000000000006100  000050f0  2**5
                  ALLOC
 26 .comment      0000001d  0000000000000000  0000000000000000  000050f0  2**0
                  CONTENTS, READONLY
 27 .debug_aranges 00000030  0000000000000000  0000000000000000  0000510d  2**0
                  CONTENTS, READONLY, DEBUGGING
 28 .debug_info   00000064  0000000000000000  0000000000000000  0000513d  2**0
                  CONTENTS, READONLY, DEBUGGING
 29 .debug_abbrev 0000004d  0000000000000000  0000000000000000  000051a1  2**0
                  CONTENTS, READONLY, DEBUGGING
 30 .debug_line   00000077  0000000000000000  0000000000000000  000051ee  2**0
                  CONTENTS, READONLY, DEBUGGING
 31 .debug_str    0000010b  0000000000000000  0000000000000000  00005265  2**0
                  CONTENTS, READONLY, DEBUGGING
 32 .debug_loc    00000059  0000000000000000  0000000000000000  00005370  2**0
                  CONTENTS, READONLY, DEBUGGING
 33 .debug_ranges 00000020  0000000000000000  0000000000000000  000053c9  2**0
                  CONTENTS, READONLY, DEBUGGING
SYMBOL TABLE:
00000000000002a8 l    d  .interp	0000000000000000              .interp
00000000000002c4 l    d  .note.ABI-tag	0000000000000000              .note.ABI-tag
00000000000002e4 l    d  .note.gnu.build-id	0000000000000000              .note.gnu.build-id
0000000000000308 l    d  .gnu.hash	0000000000000000              .gnu.hash
0000000000000338 l    d  .dynsym	0000000000000000              .dynsym
0000000000000668 l    d  .dynstr	0000000000000000              .dynstr
0000000000000a62 l    d  .gnu.version	0000000000000000              .gnu.version
0000000000000aa8 l    d  .gnu.version_r	0000000000000000              .gnu.version_r
0000000000000ba8 l    d  .rela.dyn	0000000000000000              .rela.dyn
0000000000000cf8 l    d  .rela.plt	0000000000000000              .rela.plt
0000000000001000 l    d  .init	0000000000000000              .init
0000000000001020 l    d  .plt	0000000000000000              .plt
00000000000011b0 l    d  .plt.got	0000000000000000              .plt.got
00000000000011c0 l    d  .text	0000000000000000              .text
0000000000003174 l    d  .fini	0000000000000000              .fini
0000000000004000 l    d  .rodata	0000000000000000              .rodata
00000000000041c0 l    d  .eh_frame_hdr	0000000000000000              .eh_frame_hdr
0000000000004230 l    d  .eh_frame	0000000000000000              .eh_frame
0000000000004498 l    d  .gcc_except_table	0000000000000000              .gcc_except_table
0000000000005d80 l    d  .init_array	0000000000000000              .init_array
0000000000005d98 l    d  .fini_array	0000000000000000              .fini_array
0000000000005da0 l    d  .dynamic	0000000000000000              .dynamic
0000000000005fd0 l    d  .got	0000000000000000              .got
0000000000006000 l    d  .got.plt	0000000000000000              .got.plt
00000000000060d8 l    d  .data	0000000000000000              .data
0000000000006100 l    d  .bss	0000000000000000              .bss
0000000000000000 l    d  .comment	0000000000000000              .comment
0000000000000000 l    d  .debug_aranges	0000000000000000              .debug_aranges
0000000000000000 l    d  .debug_info	0000000000000000              .debug_info
0000000000000000 l    d  .debug_abbrev	0000000000000000              .debug_abbrev
0000000000000000 l    d  .debug_line	0000000000000000              .debug_line
0000000000000000 l    d  .debug_str	0000000000000000              .debug_str
0000000000000000 l    d  .debug_loc	0000000000000000              .debug_loc
0000000000000000 l    d  .debug_ranges	0000000000000000              .debug_ranges
0000000000000000 l    df *ABS*	0000000000000000              
0000000000002820 l     F .text	00000000000001aa              _ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5
00000000000093e0 l     O .bss	0000000000000008              _ZL9generator
00000000000029d0 l     F .text	0000000000000075              _ZNSt24uniform_int_distributionIhEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEhRT_RKNS0_10param_typeE.constprop.4
0000000000002a50 l     F .text	0000000000000374              _Z9descargarPiS_._omp_fn.1
00000000000093c0 l     O .bss	0000000000000020              _ZL6zeroes
00000000000093a0 l     O .bss	0000000000000020              _ZL4ones
0000000000007380 l     O .bss	0000000000000020              _ZL8maskfff0
0000000000007360 l     O .bss	0000000000000020              _ZL8mask000f
00000000000073a0 l     O .bss	0000000000002000              _ZL4MASK
0000000000006b60 l     O .bss	0000000000000800              left_border
0000000000006360 l     O .bss	0000000000000800              right_border
0000000000002dd0 l     F .text	0000000000000321              _Z9descargarPiS_._omp_fn.0
00000000000011c0 l     F .text	0000000000000e7a              _GLOBAL__sub_I__Z8randinitv
0000000000006340 l     O .bss	0000000000000001              _ZStL8__ioinit
0000000000000000 l    df *ABS*	0000000000000000              crtfastmath.c
0000000000002710 l     F .text	0000000000000013              set_fast_math
0000000000000000 l    df *ABS*	0000000000000000              crtstuff.c
0000000000002760 l     F .text	0000000000000000              deregister_tm_clones
0000000000002790 l     F .text	0000000000000000              register_tm_clones
00000000000027d0 l     F .text	0000000000000000              __do_global_dtors_aux
0000000000006338 l     O .bss	0000000000000001              completed.7389
0000000000005d98 l     O .fini_array	0000000000000000              __do_global_dtors_aux_fini_array_entry
0000000000002810 l     F .text	0000000000000000              frame_dummy
0000000000005d80 l     O .init_array	0000000000000000              __frame_dummy_init_array_entry
0000000000000000 l    df *ABS*	0000000000000000              offloadstuff.c
0000000000000000 l    df *ABS*	0000000000000000              crtstuff.c
0000000000004494 l     O .eh_frame	0000000000000000              __FRAME_END__
0000000000000000 l    df *ABS*	0000000000000000              offloadstuff.c
0000000000000000 l    df *ABS*	0000000000000000              
00000000000041c0 l       .eh_frame_hdr	0000000000000000              __GNU_EH_FRAME_HDR
0000000000005da0 l     O .dynamic	0000000000000000              _DYNAMIC
0000000000005d98 l       .init_array	0000000000000000              __init_array_end
0000000000005d80 l       .init_array	0000000000000000              __init_array_start
0000000000006000 l     O .got.plt	0000000000000000              _GLOBAL_OFFSET_TABLE_
0000000000000000       F *UND*	0000000000000000              GOMP_single_start@@GOMP_1.0
00000000000060f0 g       .data	0000000000000000              _edata
00000000000041c0 g     O .eh_frame_hdr	0000000000000000              .hidden __offload_funcs_end
00000000000060d8  w      .data	0000000000000000              data_start
0000000000004000 g     O .rodata	0000000000000004              _IO_stdin_used
0000000000000000       F *UND*	0000000000000000              _ZNSt8ios_base15sync_with_stdioEb@@GLIBCXX_3.4
0000000000000000  w    F *UND*	0000000000000000              __cxa_finalize@@GLIBC_2.2.5
0000000000002040 g     F .text	00000000000006c8              main
0000000000000000       F *UND*	0000000000000000              GOMP_task@@GOMP_2.0
00000000000060e0 g     O .data	0000000000000000              .hidden __dso_handle
00000000000041c0 g     O .eh_frame_hdr	0000000000000000              .hidden __offload_vars_end
00000000000060e8  w    O .data	0000000000000008              .hidden DW.ref.__gxx_personality_v0
0000000000000000       F *UND*	0000000000000000              _ZNSo5flushEv@@GLIBCXX_3.4
00000000000041c0 g     O .eh_frame_hdr	0000000000000000              .hidden __offload_func_table
0000000000003174 g     F .fini	0000000000000000              _fini
0000000000000000       F *UND*	0000000000000000              aligned_alloc@@GLIBC_2.16
0000000000000000       F *UND*	0000000000000000              __cxa_atexit@@GLIBC_2.2.5
0000000000000000       F *UND*	0000000000000000              _ZNSt13random_device7_M_finiEv@@GLIBCXX_3.4.18
0000000000000000       F *UND*	0000000000000000              _ZdlPv@@GLIBCXX_3.4
0000000000002730 g     F .text	000000000000002b              _start
0000000000000000       F *UND*	0000000000000000              _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@@GLIBCXX_3.4
0000000000000000       F *UND*	0000000000000000              _Znwm@@GLIBCXX_3.4
0000000000000000       F *UND*	0000000000000000              _ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@@GLIBCXX_3.4
0000000000001000 g     F .init	0000000000000000              _init
00000000000060f0 g     O .data	0000000000000000              .hidden __TMC_END__
0000000000000000       F *UND*	0000000000000000              _ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@@GLIBCXX_3.4.9
0000000000006100 g     O .bss	0000000000000110              _ZSt4cout@@GLIBCXX_3.4
00000000000060d8 g       .data	0000000000000000              __data_start
0000000000000000       F *UND*	0000000000000000              GOMP_taskwait@@GOMP_2.0
00000000000093e8 g       .bss	0000000000000000              _end
00000000000041c0 g     O .eh_frame_hdr	0000000000000000              .hidden __offload_var_table
0000000000000000       F *UND*	0000000000000000              _ZNSt13random_device9_M_getvalEv@@GLIBCXX_3.4.18
0000000000000000       F *UND*	0000000000000000              _ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@@GLIBCXX_3.4
00000000000060f0 g       .bss	0000000000000000              __bss_start
0000000000000000       F *UND*	0000000000000000              GOMP_parallel@@GOMP_4.0
0000000000000000       F *UND*	0000000000000000              _ZNSt8ios_base4InitC1Ev@@GLIBCXX_3.4
0000000000003100 g     F .text	0000000000000065              __libc_csu_init
0000000000000000       F *UND*	0000000000000000              omp_get_thread_num@@OMP_1.0
0000000000000000       F *UND*	0000000000000000              memmove@@GLIBC_2.2.5
0000000000000000       F *UND*	0000000000000000              __gxx_personality_v0@@CXXABI_1.3
0000000000000000       F *UND*	0000000000000000              _ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@@GLIBCXX_3.4.21
0000000000000000       F *UND*	0000000000000000              _ZNSolsEi@@GLIBCXX_3.4
0000000000000000  w      *UND*	0000000000000000              _ITM_deregisterTMCloneTable
0000000000000000       F *UND*	0000000000000000              _Unwind_Resume@@GCC_3.0
0000000000003170 g     F .text	0000000000000002              __libc_csu_fini
0000000000000000       F *UND*	0000000000000000              GOMP_barrier@@GOMP_1.0
0000000000006220 g     O .bss	0000000000000118              _ZSt3cin@@GLIBCXX_3.4
0000000000000000       F *UND*	0000000000000000              __libc_start_main@@GLIBC_2.2.5
0000000000000000       F *UND*	0000000000000000              omp_get_num_threads@@OMP_1.0
0000000000000000  w      *UND*	0000000000000000              __gmon_start__
0000000000000000  w      *UND*	0000000000000000              _ITM_registerTMCloneTable
0000000000000000       F *UND*	0000000000000000              _ZNSt8ios_base4InitD1Ev@@GLIBCXX_3.4



Disassembly of section .init:

0000000000001000 <_init>:
    1000:	48 83 ec 08          	sub    $0x8,%rsp
    1004:	48 8b 05 dd 4f 00 00 	mov    0x4fdd(%rip),%rax        # 5fe8 <__gmon_start__>
    100b:	48 85 c0             	test   %rax,%rax
    100e:	74 02                	je     1012 <_init+0x12>
    1010:	ff d0                	callq  *%rax
    1012:	48 83 c4 08          	add    $0x8,%rsp
    1016:	c3                   	retq   

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	ff 35 e2 4f 00 00    	pushq  0x4fe2(%rip)        # 6008 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	ff 25 e4 4f 00 00    	jmpq   *0x4fe4(%rip)        # 6010 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000001030 <GOMP_single_start@plt>:
    1030:	ff 25 e2 4f 00 00    	jmpq   *0x4fe2(%rip)        # 6018 <GOMP_single_start@GOMP_1.0>
    1036:	68 00 00 00 00       	pushq  $0x0
    103b:	e9 e0 ff ff ff       	jmpq   1020 <.plt>

0000000000001040 <_ZNSt8ios_base15sync_with_stdioEb@plt>:
    1040:	ff 25 da 4f 00 00    	jmpq   *0x4fda(%rip)        # 6020 <_ZNSt8ios_base15sync_with_stdioEb@GLIBCXX_3.4>
    1046:	68 01 00 00 00       	pushq  $0x1
    104b:	e9 d0 ff ff ff       	jmpq   1020 <.plt>

0000000000001050 <GOMP_task@plt>:
    1050:	ff 25 d2 4f 00 00    	jmpq   *0x4fd2(%rip)        # 6028 <GOMP_task@GOMP_2.0>
    1056:	68 02 00 00 00       	pushq  $0x2
    105b:	e9 c0 ff ff ff       	jmpq   1020 <.plt>

0000000000001060 <_ZNSo5flushEv@plt>:
    1060:	ff 25 ca 4f 00 00    	jmpq   *0x4fca(%rip)        # 6030 <_ZNSo5flushEv@GLIBCXX_3.4>
    1066:	68 03 00 00 00       	pushq  $0x3
    106b:	e9 b0 ff ff ff       	jmpq   1020 <.plt>

0000000000001070 <aligned_alloc@plt>:
    1070:	ff 25 c2 4f 00 00    	jmpq   *0x4fc2(%rip)        # 6038 <aligned_alloc@GLIBC_2.16>
    1076:	68 04 00 00 00       	pushq  $0x4
    107b:	e9 a0 ff ff ff       	jmpq   1020 <.plt>

0000000000001080 <__cxa_atexit@plt>:
    1080:	ff 25 ba 4f 00 00    	jmpq   *0x4fba(%rip)        # 6040 <__cxa_atexit@GLIBC_2.2.5>
    1086:	68 05 00 00 00       	pushq  $0x5
    108b:	e9 90 ff ff ff       	jmpq   1020 <.plt>

0000000000001090 <_ZNSt13random_device7_M_finiEv@plt>:
    1090:	ff 25 b2 4f 00 00    	jmpq   *0x4fb2(%rip)        # 6048 <_ZNSt13random_device7_M_finiEv@GLIBCXX_3.4.18>
    1096:	68 06 00 00 00       	pushq  $0x6
    109b:	e9 80 ff ff ff       	jmpq   1020 <.plt>

00000000000010a0 <_ZdlPv@plt>:
    10a0:	ff 25 aa 4f 00 00    	jmpq   *0x4faa(%rip)        # 6050 <_ZdlPv@GLIBCXX_3.4>
    10a6:	68 07 00 00 00       	pushq  $0x7
    10ab:	e9 70 ff ff ff       	jmpq   1020 <.plt>

00000000000010b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>:
    10b0:	ff 25 a2 4f 00 00    	jmpq   *0x4fa2(%rip)        # 6058 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@GLIBCXX_3.4>
    10b6:	68 08 00 00 00       	pushq  $0x8
    10bb:	e9 60 ff ff ff       	jmpq   1020 <.plt>

00000000000010c0 <_Znwm@plt>:
    10c0:	ff 25 9a 4f 00 00    	jmpq   *0x4f9a(%rip)        # 6060 <_Znwm@GLIBCXX_3.4>
    10c6:	68 09 00 00 00       	pushq  $0x9
    10cb:	e9 50 ff ff ff       	jmpq   1020 <.plt>

00000000000010d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@plt>:
    10d0:	ff 25 92 4f 00 00    	jmpq   *0x4f92(%rip)        # 6068 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@GLIBCXX_3.4>
    10d6:	68 0a 00 00 00       	pushq  $0xa
    10db:	e9 40 ff ff ff       	jmpq   1020 <.plt>

00000000000010e0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>:
    10e0:	ff 25 8a 4f 00 00    	jmpq   *0x4f8a(%rip)        # 6070 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@GLIBCXX_3.4.9>
    10e6:	68 0b 00 00 00       	pushq  $0xb
    10eb:	e9 30 ff ff ff       	jmpq   1020 <.plt>

00000000000010f0 <GOMP_taskwait@plt>:
    10f0:	ff 25 82 4f 00 00    	jmpq   *0x4f82(%rip)        # 6078 <GOMP_taskwait@GOMP_2.0>
    10f6:	68 0c 00 00 00       	pushq  $0xc
    10fb:	e9 20 ff ff ff       	jmpq   1020 <.plt>

0000000000001100 <_ZNSt13random_device9_M_getvalEv@plt>:
    1100:	ff 25 7a 4f 00 00    	jmpq   *0x4f7a(%rip)        # 6080 <_ZNSt13random_device9_M_getvalEv@GLIBCXX_3.4.18>
    1106:	68 0d 00 00 00       	pushq  $0xd
    110b:	e9 10 ff ff ff       	jmpq   1020 <.plt>

0000000000001110 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>:
    1110:	ff 25 72 4f 00 00    	jmpq   *0x4f72(%rip)        # 6088 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@GLIBCXX_3.4>
    1116:	68 0e 00 00 00       	pushq  $0xe
    111b:	e9 00 ff ff ff       	jmpq   1020 <.plt>

0000000000001120 <GOMP_parallel@plt>:
    1120:	ff 25 6a 4f 00 00    	jmpq   *0x4f6a(%rip)        # 6090 <GOMP_parallel@GOMP_4.0>
    1126:	68 0f 00 00 00       	pushq  $0xf
    112b:	e9 f0 fe ff ff       	jmpq   1020 <.plt>

0000000000001130 <_ZNSt8ios_base4InitC1Ev@plt>:
    1130:	ff 25 62 4f 00 00    	jmpq   *0x4f62(%rip)        # 6098 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
    1136:	68 10 00 00 00       	pushq  $0x10
    113b:	e9 e0 fe ff ff       	jmpq   1020 <.plt>

0000000000001140 <omp_get_thread_num@plt>:
    1140:	ff 25 5a 4f 00 00    	jmpq   *0x4f5a(%rip)        # 60a0 <omp_get_thread_num@OMP_1.0>
    1146:	68 11 00 00 00       	pushq  $0x11
    114b:	e9 d0 fe ff ff       	jmpq   1020 <.plt>

0000000000001150 <memmove@plt>:
    1150:	ff 25 52 4f 00 00    	jmpq   *0x4f52(%rip)        # 60a8 <memmove@GLIBC_2.2.5>
    1156:	68 12 00 00 00       	pushq  $0x12
    115b:	e9 c0 fe ff ff       	jmpq   1020 <.plt>

0000000000001160 <_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@plt>:
    1160:	ff 25 4a 4f 00 00    	jmpq   *0x4f4a(%rip)        # 60b0 <_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@GLIBCXX_3.4.21>
    1166:	68 13 00 00 00       	pushq  $0x13
    116b:	e9 b0 fe ff ff       	jmpq   1020 <.plt>

0000000000001170 <_ZNSolsEi@plt>:
    1170:	ff 25 42 4f 00 00    	jmpq   *0x4f42(%rip)        # 60b8 <_ZNSolsEi@GLIBCXX_3.4>
    1176:	68 14 00 00 00       	pushq  $0x14
    117b:	e9 a0 fe ff ff       	jmpq   1020 <.plt>

0000000000001180 <_Unwind_Resume@plt>:
    1180:	ff 25 3a 4f 00 00    	jmpq   *0x4f3a(%rip)        # 60c0 <_Unwind_Resume@GCC_3.0>
    1186:	68 15 00 00 00       	pushq  $0x15
    118b:	e9 90 fe ff ff       	jmpq   1020 <.plt>

0000000000001190 <GOMP_barrier@plt>:
    1190:	ff 25 32 4f 00 00    	jmpq   *0x4f32(%rip)        # 60c8 <GOMP_barrier@GOMP_1.0>
    1196:	68 16 00 00 00       	pushq  $0x16
    119b:	e9 80 fe ff ff       	jmpq   1020 <.plt>

00000000000011a0 <omp_get_num_threads@plt>:
    11a0:	ff 25 2a 4f 00 00    	jmpq   *0x4f2a(%rip)        # 60d0 <omp_get_num_threads@OMP_1.0>
    11a6:	68 17 00 00 00       	pushq  $0x17
    11ab:	e9 70 fe ff ff       	jmpq   1020 <.plt>

Disassembly of section .plt.got:

00000000000011b0 <__cxa_finalize@plt>:
    11b0:	ff 25 1a 4e 00 00    	jmpq   *0x4e1a(%rip)        # 5fd0 <__cxa_finalize@GLIBC_2.2.5>
    11b6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000000011c0 <_GLOBAL__sub_I__Z8randinitv>:
    11c0:	55                   	push   %rbp
    11c1:	48 8d 3d 78 51 00 00 	lea    0x5178(%rip),%rdi        # 6340 <_ZStL8__ioinit>
    11c8:	48 89 e5             	mov    %rsp,%rbp
    11cb:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    11cf:	e8 5c ff ff ff       	callq  1130 <_ZNSt8ios_base4InitC1Ev@plt>
    11d4:	48 8b 3d 1d 4e 00 00 	mov    0x4e1d(%rip),%rdi        # 5ff8 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
    11db:	48 8d 15 fe 4e 00 00 	lea    0x4efe(%rip),%rdx        # 60e0 <__dso_handle>
    11e2:	48 8d 35 57 51 00 00 	lea    0x5157(%rip),%rsi        # 6340 <_ZStL8__ioinit>
    11e9:	e8 92 fe ff ff       	callq  1080 <__cxa_atexit@plt>
    11ee:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    11f2:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
    11f6:	48 c7 05 df 81 00 00 	movq   $0x1,0x81df(%rip)        # 93e0 <_ZL9generator>
    11fd:	01 00 00 00 
    1201:	c5 fd 7f 05 b7 81 00 	vmovdqa %ymm0,0x81b7(%rip)        # 93c0 <_ZL6zeroes>
    1208:	00 
    1209:	c5 fd 6f 05 ef 2e 00 	vmovdqa 0x2eef(%rip),%ymm0        # 4100 <_IO_stdin_used+0x100>
    1210:	00 
    1211:	c5 fd 7f 05 87 81 00 	vmovdqa %ymm0,0x8187(%rip)        # 93a0 <_ZL4ones>
    1218:	00 
    1219:	c5 fd 6f 05 ff 2e 00 	vmovdqa 0x2eff(%rip),%ymm0        # 4120 <_IO_stdin_used+0x120>
    1220:	00 
    1221:	c5 fd 7f 05 57 61 00 	vmovdqa %ymm0,0x6157(%rip)        # 7380 <_ZL8maskfff0>
    1228:	00 
    1229:	c5 fd 6f 05 0f 2f 00 	vmovdqa 0x2f0f(%rip),%ymm0        # 4140 <_IO_stdin_used+0x140>
    1230:	00 
    1231:	c5 fd 7f 05 27 61 00 	vmovdqa %ymm0,0x6127(%rip)        # 7360 <_ZL8mask000f>
    1238:	00 
    1239:	c5 fd 6f 05 bf 2e 00 	vmovdqa 0x2ebf(%rip),%ymm0        # 4100 <_IO_stdin_used+0x100>
    1240:	00 
    1241:	c4 e3 7d 02 d1 01    	vpblendd $0x1,%ymm1,%ymm0,%ymm2
    1247:	c5 fd 7f 05 51 61 00 	vmovdqa %ymm0,0x6151(%rip)        # 73a0 <_ZL4MASK>
    124e:	00 
    124f:	c5 fd 7f 15 69 61 00 	vmovdqa %ymm2,0x6169(%rip)        # 73c0 <_ZL4MASK+0x20>
    1256:	00 
    1257:	c4 e3 7d 02 d1 02    	vpblendd $0x2,%ymm1,%ymm0,%ymm2
    125d:	c5 fd 7f 15 7b 61 00 	vmovdqa %ymm2,0x617b(%rip)        # 73e0 <_ZL4MASK+0x40>
    1264:	00 
    1265:	c4 e3 7d 02 d1 03    	vpblendd $0x3,%ymm1,%ymm0,%ymm2
    126b:	c5 fd 7f 15 8d 61 00 	vmovdqa %ymm2,0x618d(%rip)        # 7400 <_ZL4MASK+0x60>
    1272:	00 
    1273:	c4 e3 7d 02 d1 04    	vpblendd $0x4,%ymm1,%ymm0,%ymm2
    1279:	c5 fd 7f 15 9f 61 00 	vmovdqa %ymm2,0x619f(%rip)        # 7420 <_ZL4MASK+0x80>
    1280:	00 
    1281:	c4 e3 7d 02 d1 05    	vpblendd $0x5,%ymm1,%ymm0,%ymm2
    1287:	c5 fd 7f 15 b1 61 00 	vmovdqa %ymm2,0x61b1(%rip)        # 7440 <_ZL4MASK+0xa0>
    128e:	00 
    128f:	c4 e3 7d 02 d1 06    	vpblendd $0x6,%ymm1,%ymm0,%ymm2
    1295:	c5 fd 7f 15 c3 61 00 	vmovdqa %ymm2,0x61c3(%rip)        # 7460 <_ZL4MASK+0xc0>
    129c:	00 
    129d:	c4 e3 7d 02 d1 07    	vpblendd $0x7,%ymm1,%ymm0,%ymm2
    12a3:	c5 fd 7f 15 d5 61 00 	vmovdqa %ymm2,0x61d5(%rip)        # 7480 <_ZL4MASK+0xe0>
    12aa:	00 
    12ab:	c4 e3 7d 02 d1 08    	vpblendd $0x8,%ymm1,%ymm0,%ymm2
    12b1:	c5 fd 7f 15 e7 61 00 	vmovdqa %ymm2,0x61e7(%rip)        # 74a0 <_ZL4MASK+0x100>
    12b8:	00 
    12b9:	c4 e3 7d 02 d1 09    	vpblendd $0x9,%ymm1,%ymm0,%ymm2
    12bf:	c5 fd 7f 15 f9 61 00 	vmovdqa %ymm2,0x61f9(%rip)        # 74c0 <_ZL4MASK+0x120>
    12c6:	00 
    12c7:	c4 e3 7d 02 d1 0a    	vpblendd $0xa,%ymm1,%ymm0,%ymm2
    12cd:	c5 fd 7f 15 0b 62 00 	vmovdqa %ymm2,0x620b(%rip)        # 74e0 <_ZL4MASK+0x140>
    12d4:	00 
    12d5:	c4 e3 7d 02 d1 0b    	vpblendd $0xb,%ymm1,%ymm0,%ymm2
    12db:	c5 fd 7f 15 1d 62 00 	vmovdqa %ymm2,0x621d(%rip)        # 7500 <_ZL4MASK+0x160>
    12e2:	00 
    12e3:	c4 e3 7d 02 d1 0c    	vpblendd $0xc,%ymm1,%ymm0,%ymm2
    12e9:	c5 fd 7f 15 2f 62 00 	vmovdqa %ymm2,0x622f(%rip)        # 7520 <_ZL4MASK+0x180>
    12f0:	00 
    12f1:	c4 e3 7d 02 d1 0d    	vpblendd $0xd,%ymm1,%ymm0,%ymm2
    12f7:	c5 fd 7f 15 41 62 00 	vmovdqa %ymm2,0x6241(%rip)        # 7540 <_ZL4MASK+0x1a0>
    12fe:	00 
    12ff:	c4 e3 7d 02 d1 0e    	vpblendd $0xe,%ymm1,%ymm0,%ymm2
    1305:	c5 fd 7f 15 53 62 00 	vmovdqa %ymm2,0x6253(%rip)        # 7560 <_ZL4MASK+0x1c0>
    130c:	00 
    130d:	c4 e3 7d 02 d1 0f    	vpblendd $0xf,%ymm1,%ymm0,%ymm2
    1313:	c5 fd 7f 15 65 62 00 	vmovdqa %ymm2,0x6265(%rip)        # 7580 <_ZL4MASK+0x1e0>
    131a:	00 
    131b:	c4 e3 7d 02 d1 10    	vpblendd $0x10,%ymm1,%ymm0,%ymm2
    1321:	c5 fd 7f 15 77 62 00 	vmovdqa %ymm2,0x6277(%rip)        # 75a0 <_ZL4MASK+0x200>
    1328:	00 
    1329:	c4 e3 7d 02 d1 11    	vpblendd $0x11,%ymm1,%ymm0,%ymm2
    132f:	c5 fd 7f 15 89 62 00 	vmovdqa %ymm2,0x6289(%rip)        # 75c0 <_ZL4MASK+0x220>
    1336:	00 
    1337:	c4 e3 7d 02 d1 12    	vpblendd $0x12,%ymm1,%ymm0,%ymm2
    133d:	c5 fd 7f 15 9b 62 00 	vmovdqa %ymm2,0x629b(%rip)        # 75e0 <_ZL4MASK+0x240>
    1344:	00 
    1345:	c4 e3 7d 02 d1 13    	vpblendd $0x13,%ymm1,%ymm0,%ymm2
    134b:	c5 fd 7f 15 ad 62 00 	vmovdqa %ymm2,0x62ad(%rip)        # 7600 <_ZL4MASK+0x260>
    1352:	00 
    1353:	c4 e3 7d 02 d1 14    	vpblendd $0x14,%ymm1,%ymm0,%ymm2
    1359:	c5 fd 7f 15 bf 62 00 	vmovdqa %ymm2,0x62bf(%rip)        # 7620 <_ZL4MASK+0x280>
    1360:	00 
    1361:	c4 e3 7d 02 d1 15    	vpblendd $0x15,%ymm1,%ymm0,%ymm2
    1367:	c5 fd 7f 15 d1 62 00 	vmovdqa %ymm2,0x62d1(%rip)        # 7640 <_ZL4MASK+0x2a0>
    136e:	00 
    136f:	c4 e3 7d 02 d1 16    	vpblendd $0x16,%ymm1,%ymm0,%ymm2
    1375:	c5 fd 7f 15 e3 62 00 	vmovdqa %ymm2,0x62e3(%rip)        # 7660 <_ZL4MASK+0x2c0>
    137c:	00 
    137d:	c4 e3 7d 02 d1 17    	vpblendd $0x17,%ymm1,%ymm0,%ymm2
    1383:	c5 fd 7f 15 f5 62 00 	vmovdqa %ymm2,0x62f5(%rip)        # 7680 <_ZL4MASK+0x2e0>
    138a:	00 
    138b:	c4 e3 7d 02 d1 18    	vpblendd $0x18,%ymm1,%ymm0,%ymm2
    1391:	c5 fd 7f 15 07 63 00 	vmovdqa %ymm2,0x6307(%rip)        # 76a0 <_ZL4MASK+0x300>
    1398:	00 
    1399:	c4 e3 7d 02 d1 19    	vpblendd $0x19,%ymm1,%ymm0,%ymm2
    139f:	c5 fd 7f 15 19 63 00 	vmovdqa %ymm2,0x6319(%rip)        # 76c0 <_ZL4MASK+0x320>
    13a6:	00 
    13a7:	c4 e3 7d 02 d1 1a    	vpblendd $0x1a,%ymm1,%ymm0,%ymm2
    13ad:	c5 fd 7f 15 2b 63 00 	vmovdqa %ymm2,0x632b(%rip)        # 76e0 <_ZL4MASK+0x340>
    13b4:	00 
    13b5:	c4 e3 7d 02 d1 1b    	vpblendd $0x1b,%ymm1,%ymm0,%ymm2
    13bb:	c5 fd 7f 15 3d 63 00 	vmovdqa %ymm2,0x633d(%rip)        # 7700 <_ZL4MASK+0x360>
    13c2:	00 
    13c3:	c4 e3 7d 02 d1 1c    	vpblendd $0x1c,%ymm1,%ymm0,%ymm2
    13c9:	c5 fd 7f 15 4f 63 00 	vmovdqa %ymm2,0x634f(%rip)        # 7720 <_ZL4MASK+0x380>
    13d0:	00 
    13d1:	c4 e3 7d 02 d1 1d    	vpblendd $0x1d,%ymm1,%ymm0,%ymm2
    13d7:	c5 fd 7f 15 61 63 00 	vmovdqa %ymm2,0x6361(%rip)        # 7740 <_ZL4MASK+0x3a0>
    13de:	00 
    13df:	c4 e3 7d 02 d1 1e    	vpblendd $0x1e,%ymm1,%ymm0,%ymm2
    13e5:	c5 fd 7f 15 73 63 00 	vmovdqa %ymm2,0x6373(%rip)        # 7760 <_ZL4MASK+0x3c0>
    13ec:	00 
    13ed:	c4 e3 7d 02 d1 1f    	vpblendd $0x1f,%ymm1,%ymm0,%ymm2
    13f3:	c5 fd 7f 15 85 63 00 	vmovdqa %ymm2,0x6385(%rip)        # 7780 <_ZL4MASK+0x3e0>
    13fa:	00 
    13fb:	c4 e3 7d 02 d1 20    	vpblendd $0x20,%ymm1,%ymm0,%ymm2
    1401:	c5 fd 7f 15 97 63 00 	vmovdqa %ymm2,0x6397(%rip)        # 77a0 <_ZL4MASK+0x400>
    1408:	00 
    1409:	c4 e3 7d 02 d1 21    	vpblendd $0x21,%ymm1,%ymm0,%ymm2
    140f:	c5 fd 7f 15 a9 63 00 	vmovdqa %ymm2,0x63a9(%rip)        # 77c0 <_ZL4MASK+0x420>
    1416:	00 
    1417:	c4 e3 7d 02 d1 22    	vpblendd $0x22,%ymm1,%ymm0,%ymm2
    141d:	c5 fd 7f 15 bb 63 00 	vmovdqa %ymm2,0x63bb(%rip)        # 77e0 <_ZL4MASK+0x440>
    1424:	00 
    1425:	c4 e3 7d 02 d1 23    	vpblendd $0x23,%ymm1,%ymm0,%ymm2
    142b:	c5 fd 7f 15 cd 63 00 	vmovdqa %ymm2,0x63cd(%rip)        # 7800 <_ZL4MASK+0x460>
    1432:	00 
    1433:	c4 e3 7d 02 d1 24    	vpblendd $0x24,%ymm1,%ymm0,%ymm2
    1439:	c5 fd 7f 15 df 63 00 	vmovdqa %ymm2,0x63df(%rip)        # 7820 <_ZL4MASK+0x480>
    1440:	00 
    1441:	c4 e3 7d 02 d1 25    	vpblendd $0x25,%ymm1,%ymm0,%ymm2
    1447:	c5 fd 7f 15 f1 63 00 	vmovdqa %ymm2,0x63f1(%rip)        # 7840 <_ZL4MASK+0x4a0>
    144e:	00 
    144f:	c4 e3 7d 02 d1 26    	vpblendd $0x26,%ymm1,%ymm0,%ymm2
    1455:	c5 fd 7f 15 03 64 00 	vmovdqa %ymm2,0x6403(%rip)        # 7860 <_ZL4MASK+0x4c0>
    145c:	00 
    145d:	c4 e3 7d 02 d1 27    	vpblendd $0x27,%ymm1,%ymm0,%ymm2
    1463:	c5 fd 7f 15 15 64 00 	vmovdqa %ymm2,0x6415(%rip)        # 7880 <_ZL4MASK+0x4e0>
    146a:	00 
    146b:	c4 e3 7d 02 d1 28    	vpblendd $0x28,%ymm1,%ymm0,%ymm2
    1471:	c5 fd 7f 15 27 64 00 	vmovdqa %ymm2,0x6427(%rip)        # 78a0 <_ZL4MASK+0x500>
    1478:	00 
    1479:	c4 e3 7d 02 d1 29    	vpblendd $0x29,%ymm1,%ymm0,%ymm2
    147f:	c5 fd 7f 15 39 64 00 	vmovdqa %ymm2,0x6439(%rip)        # 78c0 <_ZL4MASK+0x520>
    1486:	00 
    1487:	c4 e3 7d 02 d1 2a    	vpblendd $0x2a,%ymm1,%ymm0,%ymm2
    148d:	c5 fd 7f 15 4b 64 00 	vmovdqa %ymm2,0x644b(%rip)        # 78e0 <_ZL4MASK+0x540>
    1494:	00 
    1495:	c4 e3 7d 02 d1 2b    	vpblendd $0x2b,%ymm1,%ymm0,%ymm2
    149b:	c5 fd 7f 15 5d 64 00 	vmovdqa %ymm2,0x645d(%rip)        # 7900 <_ZL4MASK+0x560>
    14a2:	00 
    14a3:	c4 e3 7d 02 d1 2c    	vpblendd $0x2c,%ymm1,%ymm0,%ymm2
    14a9:	c5 fd 7f 15 6f 64 00 	vmovdqa %ymm2,0x646f(%rip)        # 7920 <_ZL4MASK+0x580>
    14b0:	00 
    14b1:	c4 e3 7d 02 d1 2d    	vpblendd $0x2d,%ymm1,%ymm0,%ymm2
    14b7:	c5 fd 7f 15 81 64 00 	vmovdqa %ymm2,0x6481(%rip)        # 7940 <_ZL4MASK+0x5a0>
    14be:	00 
    14bf:	c4 e3 7d 02 d1 2e    	vpblendd $0x2e,%ymm1,%ymm0,%ymm2
    14c5:	c5 fd 7f 15 93 64 00 	vmovdqa %ymm2,0x6493(%rip)        # 7960 <_ZL4MASK+0x5c0>
    14cc:	00 
    14cd:	c4 e3 7d 02 d1 2f    	vpblendd $0x2f,%ymm1,%ymm0,%ymm2
    14d3:	c5 fd 7f 15 a5 64 00 	vmovdqa %ymm2,0x64a5(%rip)        # 7980 <_ZL4MASK+0x5e0>
    14da:	00 
    14db:	c4 e3 7d 02 d1 30    	vpblendd $0x30,%ymm1,%ymm0,%ymm2
    14e1:	c5 fd 7f 15 b7 64 00 	vmovdqa %ymm2,0x64b7(%rip)        # 79a0 <_ZL4MASK+0x600>
    14e8:	00 
    14e9:	c4 e3 7d 02 d1 31    	vpblendd $0x31,%ymm1,%ymm0,%ymm2
    14ef:	c5 fd 7f 15 c9 64 00 	vmovdqa %ymm2,0x64c9(%rip)        # 79c0 <_ZL4MASK+0x620>
    14f6:	00 
    14f7:	c4 e3 7d 02 d1 32    	vpblendd $0x32,%ymm1,%ymm0,%ymm2
    14fd:	c5 fd 7f 15 db 64 00 	vmovdqa %ymm2,0x64db(%rip)        # 79e0 <_ZL4MASK+0x640>
    1504:	00 
    1505:	c4 e3 7d 02 d1 33    	vpblendd $0x33,%ymm1,%ymm0,%ymm2
    150b:	c5 fd 7f 15 ed 64 00 	vmovdqa %ymm2,0x64ed(%rip)        # 7a00 <_ZL4MASK+0x660>
    1512:	00 
    1513:	c4 e3 7d 02 d1 34    	vpblendd $0x34,%ymm1,%ymm0,%ymm2
    1519:	c5 fd 7f 15 ff 64 00 	vmovdqa %ymm2,0x64ff(%rip)        # 7a20 <_ZL4MASK+0x680>
    1520:	00 
    1521:	c4 e3 7d 02 d1 35    	vpblendd $0x35,%ymm1,%ymm0,%ymm2
    1527:	c5 fd 7f 15 11 65 00 	vmovdqa %ymm2,0x6511(%rip)        # 7a40 <_ZL4MASK+0x6a0>
    152e:	00 
    152f:	c4 e3 7d 02 d1 36    	vpblendd $0x36,%ymm1,%ymm0,%ymm2
    1535:	c5 fd 7f 15 23 65 00 	vmovdqa %ymm2,0x6523(%rip)        # 7a60 <_ZL4MASK+0x6c0>
    153c:	00 
    153d:	c4 e3 7d 02 d1 37    	vpblendd $0x37,%ymm1,%ymm0,%ymm2
    1543:	c5 fd 7f 15 35 65 00 	vmovdqa %ymm2,0x6535(%rip)        # 7a80 <_ZL4MASK+0x6e0>
    154a:	00 
    154b:	c4 e3 7d 02 d1 38    	vpblendd $0x38,%ymm1,%ymm0,%ymm2
    1551:	c5 fd 7f 15 47 65 00 	vmovdqa %ymm2,0x6547(%rip)        # 7aa0 <_ZL4MASK+0x700>
    1558:	00 
    1559:	c4 e3 7d 02 d1 39    	vpblendd $0x39,%ymm1,%ymm0,%ymm2
    155f:	c5 fd 7f 15 59 65 00 	vmovdqa %ymm2,0x6559(%rip)        # 7ac0 <_ZL4MASK+0x720>
    1566:	00 
    1567:	c4 e3 7d 02 d1 3a    	vpblendd $0x3a,%ymm1,%ymm0,%ymm2
    156d:	c5 fd 7f 15 6b 65 00 	vmovdqa %ymm2,0x656b(%rip)        # 7ae0 <_ZL4MASK+0x740>
    1574:	00 
    1575:	c4 e3 7d 02 d1 3b    	vpblendd $0x3b,%ymm1,%ymm0,%ymm2
    157b:	c5 fd 7f 15 7d 65 00 	vmovdqa %ymm2,0x657d(%rip)        # 7b00 <_ZL4MASK+0x760>
    1582:	00 
    1583:	c4 e3 7d 02 d1 3c    	vpblendd $0x3c,%ymm1,%ymm0,%ymm2
    1589:	c5 fd 7f 15 8f 65 00 	vmovdqa %ymm2,0x658f(%rip)        # 7b20 <_ZL4MASK+0x780>
    1590:	00 
    1591:	c4 e3 7d 02 d1 3d    	vpblendd $0x3d,%ymm1,%ymm0,%ymm2
    1597:	c5 fd 7f 15 a1 65 00 	vmovdqa %ymm2,0x65a1(%rip)        # 7b40 <_ZL4MASK+0x7a0>
    159e:	00 
    159f:	c4 e3 7d 02 d1 3e    	vpblendd $0x3e,%ymm1,%ymm0,%ymm2
    15a5:	c5 fd 7f 15 b3 65 00 	vmovdqa %ymm2,0x65b3(%rip)        # 7b60 <_ZL4MASK+0x7c0>
    15ac:	00 
    15ad:	c4 e3 7d 02 d1 3f    	vpblendd $0x3f,%ymm1,%ymm0,%ymm2
    15b3:	c5 fd 7f 15 c5 65 00 	vmovdqa %ymm2,0x65c5(%rip)        # 7b80 <_ZL4MASK+0x7e0>
    15ba:	00 
    15bb:	c4 e3 7d 02 d1 40    	vpblendd $0x40,%ymm1,%ymm0,%ymm2
    15c1:	c5 fd 7f 15 d7 65 00 	vmovdqa %ymm2,0x65d7(%rip)        # 7ba0 <_ZL4MASK+0x800>
    15c8:	00 
    15c9:	c4 e3 7d 02 d1 41    	vpblendd $0x41,%ymm1,%ymm0,%ymm2
    15cf:	c5 fd 7f 15 e9 65 00 	vmovdqa %ymm2,0x65e9(%rip)        # 7bc0 <_ZL4MASK+0x820>
    15d6:	00 
    15d7:	c4 e3 7d 02 d1 42    	vpblendd $0x42,%ymm1,%ymm0,%ymm2
    15dd:	c5 fd 7f 15 fb 65 00 	vmovdqa %ymm2,0x65fb(%rip)        # 7be0 <_ZL4MASK+0x840>
    15e4:	00 
    15e5:	c4 e3 7d 02 d1 43    	vpblendd $0x43,%ymm1,%ymm0,%ymm2
    15eb:	c5 fd 7f 15 0d 66 00 	vmovdqa %ymm2,0x660d(%rip)        # 7c00 <_ZL4MASK+0x860>
    15f2:	00 
    15f3:	c4 e3 7d 02 d1 44    	vpblendd $0x44,%ymm1,%ymm0,%ymm2
    15f9:	c5 fd 7f 15 1f 66 00 	vmovdqa %ymm2,0x661f(%rip)        # 7c20 <_ZL4MASK+0x880>
    1600:	00 
    1601:	c4 e3 7d 02 d1 45    	vpblendd $0x45,%ymm1,%ymm0,%ymm2
    1607:	c5 fd 7f 15 31 66 00 	vmovdqa %ymm2,0x6631(%rip)        # 7c40 <_ZL4MASK+0x8a0>
    160e:	00 
    160f:	c4 e3 7d 02 d1 46    	vpblendd $0x46,%ymm1,%ymm0,%ymm2
    1615:	c5 fd 7f 15 43 66 00 	vmovdqa %ymm2,0x6643(%rip)        # 7c60 <_ZL4MASK+0x8c0>
    161c:	00 
    161d:	c4 e3 7d 02 d1 47    	vpblendd $0x47,%ymm1,%ymm0,%ymm2
    1623:	c5 fd 7f 15 55 66 00 	vmovdqa %ymm2,0x6655(%rip)        # 7c80 <_ZL4MASK+0x8e0>
    162a:	00 
    162b:	c4 e3 7d 02 d1 48    	vpblendd $0x48,%ymm1,%ymm0,%ymm2
    1631:	c5 fd 7f 15 67 66 00 	vmovdqa %ymm2,0x6667(%rip)        # 7ca0 <_ZL4MASK+0x900>
    1638:	00 
    1639:	c4 e3 7d 02 d1 49    	vpblendd $0x49,%ymm1,%ymm0,%ymm2
    163f:	c5 fd 7f 15 79 66 00 	vmovdqa %ymm2,0x6679(%rip)        # 7cc0 <_ZL4MASK+0x920>
    1646:	00 
    1647:	c4 e3 7d 02 d1 4a    	vpblendd $0x4a,%ymm1,%ymm0,%ymm2
    164d:	c5 fd 7f 15 8b 66 00 	vmovdqa %ymm2,0x668b(%rip)        # 7ce0 <_ZL4MASK+0x940>
    1654:	00 
    1655:	c4 e3 7d 02 d1 4b    	vpblendd $0x4b,%ymm1,%ymm0,%ymm2
    165b:	c5 fd 7f 15 9d 66 00 	vmovdqa %ymm2,0x669d(%rip)        # 7d00 <_ZL4MASK+0x960>
    1662:	00 
    1663:	c4 e3 7d 02 d1 4c    	vpblendd $0x4c,%ymm1,%ymm0,%ymm2
    1669:	c5 fd 7f 15 af 66 00 	vmovdqa %ymm2,0x66af(%rip)        # 7d20 <_ZL4MASK+0x980>
    1670:	00 
    1671:	c4 e3 7d 02 d1 4d    	vpblendd $0x4d,%ymm1,%ymm0,%ymm2
    1677:	c5 fd 7f 15 c1 66 00 	vmovdqa %ymm2,0x66c1(%rip)        # 7d40 <_ZL4MASK+0x9a0>
    167e:	00 
    167f:	c4 e3 7d 02 d1 4e    	vpblendd $0x4e,%ymm1,%ymm0,%ymm2
    1685:	c5 fd 7f 15 d3 66 00 	vmovdqa %ymm2,0x66d3(%rip)        # 7d60 <_ZL4MASK+0x9c0>
    168c:	00 
    168d:	c4 e3 7d 02 d1 4f    	vpblendd $0x4f,%ymm1,%ymm0,%ymm2
    1693:	c5 fd 7f 15 e5 66 00 	vmovdqa %ymm2,0x66e5(%rip)        # 7d80 <_ZL4MASK+0x9e0>
    169a:	00 
    169b:	c4 e3 7d 02 d1 50    	vpblendd $0x50,%ymm1,%ymm0,%ymm2
    16a1:	c5 fd 7f 15 f7 66 00 	vmovdqa %ymm2,0x66f7(%rip)        # 7da0 <_ZL4MASK+0xa00>
    16a8:	00 
    16a9:	c4 e3 7d 02 d1 51    	vpblendd $0x51,%ymm1,%ymm0,%ymm2
    16af:	c5 fd 7f 15 09 67 00 	vmovdqa %ymm2,0x6709(%rip)        # 7dc0 <_ZL4MASK+0xa20>
    16b6:	00 
    16b7:	c4 e3 7d 02 d1 52    	vpblendd $0x52,%ymm1,%ymm0,%ymm2
    16bd:	c5 fd 7f 15 1b 67 00 	vmovdqa %ymm2,0x671b(%rip)        # 7de0 <_ZL4MASK+0xa40>
    16c4:	00 
    16c5:	c4 e3 7d 02 d1 53    	vpblendd $0x53,%ymm1,%ymm0,%ymm2
    16cb:	c5 fd 7f 15 2d 67 00 	vmovdqa %ymm2,0x672d(%rip)        # 7e00 <_ZL4MASK+0xa60>
    16d2:	00 
    16d3:	c4 e3 7d 02 d1 54    	vpblendd $0x54,%ymm1,%ymm0,%ymm2
    16d9:	c5 fd 7f 15 3f 67 00 	vmovdqa %ymm2,0x673f(%rip)        # 7e20 <_ZL4MASK+0xa80>
    16e0:	00 
    16e1:	c4 e3 7d 02 d1 55    	vpblendd $0x55,%ymm1,%ymm0,%ymm2
    16e7:	c5 fd 7f 15 51 67 00 	vmovdqa %ymm2,0x6751(%rip)        # 7e40 <_ZL4MASK+0xaa0>
    16ee:	00 
    16ef:	c4 e3 7d 02 d1 56    	vpblendd $0x56,%ymm1,%ymm0,%ymm2
    16f5:	c5 fd 7f 15 63 67 00 	vmovdqa %ymm2,0x6763(%rip)        # 7e60 <_ZL4MASK+0xac0>
    16fc:	00 
    16fd:	c4 e3 7d 02 d1 57    	vpblendd $0x57,%ymm1,%ymm0,%ymm2
    1703:	c5 fd 7f 15 75 67 00 	vmovdqa %ymm2,0x6775(%rip)        # 7e80 <_ZL4MASK+0xae0>
    170a:	00 
    170b:	c4 e3 7d 02 d1 58    	vpblendd $0x58,%ymm1,%ymm0,%ymm2
    1711:	c5 fd 7f 15 87 67 00 	vmovdqa %ymm2,0x6787(%rip)        # 7ea0 <_ZL4MASK+0xb00>
    1718:	00 
    1719:	c4 e3 7d 02 d1 59    	vpblendd $0x59,%ymm1,%ymm0,%ymm2
    171f:	c5 fd 7f 15 99 67 00 	vmovdqa %ymm2,0x6799(%rip)        # 7ec0 <_ZL4MASK+0xb20>
    1726:	00 
    1727:	c4 e3 7d 02 d1 5a    	vpblendd $0x5a,%ymm1,%ymm0,%ymm2
    172d:	c5 fd 7f 15 ab 67 00 	vmovdqa %ymm2,0x67ab(%rip)        # 7ee0 <_ZL4MASK+0xb40>
    1734:	00 
    1735:	c4 e3 7d 02 d1 5b    	vpblendd $0x5b,%ymm1,%ymm0,%ymm2
    173b:	c5 fd 7f 15 bd 67 00 	vmovdqa %ymm2,0x67bd(%rip)        # 7f00 <_ZL4MASK+0xb60>
    1742:	00 
    1743:	c4 e3 7d 02 d1 5c    	vpblendd $0x5c,%ymm1,%ymm0,%ymm2
    1749:	c5 fd 7f 15 cf 67 00 	vmovdqa %ymm2,0x67cf(%rip)        # 7f20 <_ZL4MASK+0xb80>
    1750:	00 
    1751:	c4 e3 7d 02 d1 5d    	vpblendd $0x5d,%ymm1,%ymm0,%ymm2
    1757:	c5 fd 7f 15 e1 67 00 	vmovdqa %ymm2,0x67e1(%rip)        # 7f40 <_ZL4MASK+0xba0>
    175e:	00 
    175f:	c4 e3 7d 02 d1 5e    	vpblendd $0x5e,%ymm1,%ymm0,%ymm2
    1765:	c5 fd 7f 15 f3 67 00 	vmovdqa %ymm2,0x67f3(%rip)        # 7f60 <_ZL4MASK+0xbc0>
    176c:	00 
    176d:	c4 e3 7d 02 d1 5f    	vpblendd $0x5f,%ymm1,%ymm0,%ymm2
    1773:	c5 fd 7f 15 05 68 00 	vmovdqa %ymm2,0x6805(%rip)        # 7f80 <_ZL4MASK+0xbe0>
    177a:	00 
    177b:	c4 e3 7d 02 d1 60    	vpblendd $0x60,%ymm1,%ymm0,%ymm2
    1781:	c5 fd 7f 15 17 68 00 	vmovdqa %ymm2,0x6817(%rip)        # 7fa0 <_ZL4MASK+0xc00>
    1788:	00 
    1789:	c4 e3 7d 02 d1 61    	vpblendd $0x61,%ymm1,%ymm0,%ymm2
    178f:	c5 fd 7f 15 29 68 00 	vmovdqa %ymm2,0x6829(%rip)        # 7fc0 <_ZL4MASK+0xc20>
    1796:	00 
    1797:	c4 e3 7d 02 d1 62    	vpblendd $0x62,%ymm1,%ymm0,%ymm2
    179d:	c5 fd 7f 15 3b 68 00 	vmovdqa %ymm2,0x683b(%rip)        # 7fe0 <_ZL4MASK+0xc40>
    17a4:	00 
    17a5:	c4 e3 7d 02 d1 63    	vpblendd $0x63,%ymm1,%ymm0,%ymm2
    17ab:	c5 fd 7f 15 4d 68 00 	vmovdqa %ymm2,0x684d(%rip)        # 8000 <_ZL4MASK+0xc60>
    17b2:	00 
    17b3:	c4 e3 7d 02 d1 64    	vpblendd $0x64,%ymm1,%ymm0,%ymm2
    17b9:	c5 fd 7f 15 5f 68 00 	vmovdqa %ymm2,0x685f(%rip)        # 8020 <_ZL4MASK+0xc80>
    17c0:	00 
    17c1:	c4 e3 7d 02 d1 65    	vpblendd $0x65,%ymm1,%ymm0,%ymm2
    17c7:	c5 fd 7f 15 71 68 00 	vmovdqa %ymm2,0x6871(%rip)        # 8040 <_ZL4MASK+0xca0>
    17ce:	00 
    17cf:	c4 e3 7d 02 d1 66    	vpblendd $0x66,%ymm1,%ymm0,%ymm2
    17d5:	c5 fd 7f 15 83 68 00 	vmovdqa %ymm2,0x6883(%rip)        # 8060 <_ZL4MASK+0xcc0>
    17dc:	00 
    17dd:	c4 e3 7d 02 d1 67    	vpblendd $0x67,%ymm1,%ymm0,%ymm2
    17e3:	c5 fd 7f 15 95 68 00 	vmovdqa %ymm2,0x6895(%rip)        # 8080 <_ZL4MASK+0xce0>
    17ea:	00 
    17eb:	c4 e3 7d 02 d1 68    	vpblendd $0x68,%ymm1,%ymm0,%ymm2
    17f1:	c5 fd 7f 15 a7 68 00 	vmovdqa %ymm2,0x68a7(%rip)        # 80a0 <_ZL4MASK+0xd00>
    17f8:	00 
    17f9:	c4 e3 7d 02 d1 69    	vpblendd $0x69,%ymm1,%ymm0,%ymm2
    17ff:	c5 fd 7f 15 b9 68 00 	vmovdqa %ymm2,0x68b9(%rip)        # 80c0 <_ZL4MASK+0xd20>
    1806:	00 
    1807:	c4 e3 7d 02 d1 6a    	vpblendd $0x6a,%ymm1,%ymm0,%ymm2
    180d:	c5 fd 7f 15 cb 68 00 	vmovdqa %ymm2,0x68cb(%rip)        # 80e0 <_ZL4MASK+0xd40>
    1814:	00 
    1815:	c4 e3 7d 02 d1 6b    	vpblendd $0x6b,%ymm1,%ymm0,%ymm2
    181b:	c5 fd 7f 15 dd 68 00 	vmovdqa %ymm2,0x68dd(%rip)        # 8100 <_ZL4MASK+0xd60>
    1822:	00 
    1823:	c4 e3 7d 02 d1 6c    	vpblendd $0x6c,%ymm1,%ymm0,%ymm2
    1829:	c5 fd 7f 15 ef 68 00 	vmovdqa %ymm2,0x68ef(%rip)        # 8120 <_ZL4MASK+0xd80>
    1830:	00 
    1831:	c4 e3 7d 02 d1 6d    	vpblendd $0x6d,%ymm1,%ymm0,%ymm2
    1837:	c5 fd 7f 15 01 69 00 	vmovdqa %ymm2,0x6901(%rip)        # 8140 <_ZL4MASK+0xda0>
    183e:	00 
    183f:	c4 e3 7d 02 d1 6e    	vpblendd $0x6e,%ymm1,%ymm0,%ymm2
    1845:	c5 fd 7f 15 13 69 00 	vmovdqa %ymm2,0x6913(%rip)        # 8160 <_ZL4MASK+0xdc0>
    184c:	00 
    184d:	c4 e3 7d 02 d1 6f    	vpblendd $0x6f,%ymm1,%ymm0,%ymm2
    1853:	c5 fd 7f 15 25 69 00 	vmovdqa %ymm2,0x6925(%rip)        # 8180 <_ZL4MASK+0xde0>
    185a:	00 
    185b:	c4 e3 7d 02 d1 70    	vpblendd $0x70,%ymm1,%ymm0,%ymm2
    1861:	c5 fd 7f 15 37 69 00 	vmovdqa %ymm2,0x6937(%rip)        # 81a0 <_ZL4MASK+0xe00>
    1868:	00 
    1869:	c4 e3 7d 02 d1 71    	vpblendd $0x71,%ymm1,%ymm0,%ymm2
    186f:	c5 fd 7f 15 49 69 00 	vmovdqa %ymm2,0x6949(%rip)        # 81c0 <_ZL4MASK+0xe20>
    1876:	00 
    1877:	c4 e3 7d 02 d1 72    	vpblendd $0x72,%ymm1,%ymm0,%ymm2
    187d:	c5 fd 7f 15 5b 69 00 	vmovdqa %ymm2,0x695b(%rip)        # 81e0 <_ZL4MASK+0xe40>
    1884:	00 
    1885:	c4 e3 7d 02 d1 73    	vpblendd $0x73,%ymm1,%ymm0,%ymm2
    188b:	c5 fd 7f 15 6d 69 00 	vmovdqa %ymm2,0x696d(%rip)        # 8200 <_ZL4MASK+0xe60>
    1892:	00 
    1893:	c4 e3 7d 02 d1 74    	vpblendd $0x74,%ymm1,%ymm0,%ymm2
    1899:	c5 fd 7f 15 7f 69 00 	vmovdqa %ymm2,0x697f(%rip)        # 8220 <_ZL4MASK+0xe80>
    18a0:	00 
    18a1:	c4 e3 7d 02 d1 75    	vpblendd $0x75,%ymm1,%ymm0,%ymm2
    18a7:	c5 fd 7f 15 91 69 00 	vmovdqa %ymm2,0x6991(%rip)        # 8240 <_ZL4MASK+0xea0>
    18ae:	00 
    18af:	c4 e3 7d 02 d1 76    	vpblendd $0x76,%ymm1,%ymm0,%ymm2
    18b5:	c5 fd 7f 15 a3 69 00 	vmovdqa %ymm2,0x69a3(%rip)        # 8260 <_ZL4MASK+0xec0>
    18bc:	00 
    18bd:	c4 e3 7d 02 d1 77    	vpblendd $0x77,%ymm1,%ymm0,%ymm2
    18c3:	c5 fd 7f 15 b5 69 00 	vmovdqa %ymm2,0x69b5(%rip)        # 8280 <_ZL4MASK+0xee0>
    18ca:	00 
    18cb:	c4 e3 7d 02 d1 78    	vpblendd $0x78,%ymm1,%ymm0,%ymm2
    18d1:	c5 fd 7f 15 c7 69 00 	vmovdqa %ymm2,0x69c7(%rip)        # 82a0 <_ZL4MASK+0xf00>
    18d8:	00 
    18d9:	c4 e3 7d 02 d1 79    	vpblendd $0x79,%ymm1,%ymm0,%ymm2
    18df:	c5 fd 7f 15 d9 69 00 	vmovdqa %ymm2,0x69d9(%rip)        # 82c0 <_ZL4MASK+0xf20>
    18e6:	00 
    18e7:	c4 e3 7d 02 d1 7a    	vpblendd $0x7a,%ymm1,%ymm0,%ymm2
    18ed:	c5 fd 7f 15 eb 69 00 	vmovdqa %ymm2,0x69eb(%rip)        # 82e0 <_ZL4MASK+0xf40>
    18f4:	00 
    18f5:	c4 e3 7d 02 d1 7b    	vpblendd $0x7b,%ymm1,%ymm0,%ymm2
    18fb:	c5 fd 7f 15 fd 69 00 	vmovdqa %ymm2,0x69fd(%rip)        # 8300 <_ZL4MASK+0xf60>
    1902:	00 
    1903:	c4 e3 7d 02 d1 7c    	vpblendd $0x7c,%ymm1,%ymm0,%ymm2
    1909:	c5 fd 7f 15 0f 6a 00 	vmovdqa %ymm2,0x6a0f(%rip)        # 8320 <_ZL4MASK+0xf80>
    1910:	00 
    1911:	c4 e3 7d 02 d1 7d    	vpblendd $0x7d,%ymm1,%ymm0,%ymm2
    1917:	c5 fd 7f 15 21 6a 00 	vmovdqa %ymm2,0x6a21(%rip)        # 8340 <_ZL4MASK+0xfa0>
    191e:	00 
    191f:	c4 e3 7d 02 d1 7e    	vpblendd $0x7e,%ymm1,%ymm0,%ymm2
    1925:	c5 fd 7f 15 33 6a 00 	vmovdqa %ymm2,0x6a33(%rip)        # 8360 <_ZL4MASK+0xfc0>
    192c:	00 
    192d:	c4 e3 7d 02 d1 7f    	vpblendd $0x7f,%ymm1,%ymm0,%ymm2
    1933:	c5 fd 7f 15 45 6a 00 	vmovdqa %ymm2,0x6a45(%rip)        # 8380 <_ZL4MASK+0xfe0>
    193a:	00 
    193b:	c4 e3 7d 02 d1 80    	vpblendd $0x80,%ymm1,%ymm0,%ymm2
    1941:	c5 fd 7f 15 57 6a 00 	vmovdqa %ymm2,0x6a57(%rip)        # 83a0 <_ZL4MASK+0x1000>
    1948:	00 
    1949:	c4 e3 7d 02 d1 81    	vpblendd $0x81,%ymm1,%ymm0,%ymm2
    194f:	c5 fd 7f 15 69 6a 00 	vmovdqa %ymm2,0x6a69(%rip)        # 83c0 <_ZL4MASK+0x1020>
    1956:	00 
    1957:	c4 e3 7d 02 d1 82    	vpblendd $0x82,%ymm1,%ymm0,%ymm2
    195d:	c5 fd 7f 15 7b 6a 00 	vmovdqa %ymm2,0x6a7b(%rip)        # 83e0 <_ZL4MASK+0x1040>
    1964:	00 
    1965:	c4 e3 7d 02 d1 83    	vpblendd $0x83,%ymm1,%ymm0,%ymm2
    196b:	c5 fd 7f 15 8d 6a 00 	vmovdqa %ymm2,0x6a8d(%rip)        # 8400 <_ZL4MASK+0x1060>
    1972:	00 
    1973:	c4 e3 7d 02 d1 84    	vpblendd $0x84,%ymm1,%ymm0,%ymm2
    1979:	c5 fd 7f 15 9f 6a 00 	vmovdqa %ymm2,0x6a9f(%rip)        # 8420 <_ZL4MASK+0x1080>
    1980:	00 
    1981:	c4 e3 7d 02 d1 85    	vpblendd $0x85,%ymm1,%ymm0,%ymm2
    1987:	c5 fd 7f 15 b1 6a 00 	vmovdqa %ymm2,0x6ab1(%rip)        # 8440 <_ZL4MASK+0x10a0>
    198e:	00 
    198f:	c4 e3 7d 02 d1 86    	vpblendd $0x86,%ymm1,%ymm0,%ymm2
    1995:	c5 fd 7f 15 c3 6a 00 	vmovdqa %ymm2,0x6ac3(%rip)        # 8460 <_ZL4MASK+0x10c0>
    199c:	00 
    199d:	c4 e3 7d 02 d1 87    	vpblendd $0x87,%ymm1,%ymm0,%ymm2
    19a3:	c5 fd 7f 15 d5 6a 00 	vmovdqa %ymm2,0x6ad5(%rip)        # 8480 <_ZL4MASK+0x10e0>
    19aa:	00 
    19ab:	c4 e3 7d 02 d1 88    	vpblendd $0x88,%ymm1,%ymm0,%ymm2
    19b1:	c5 fd 7f 15 e7 6a 00 	vmovdqa %ymm2,0x6ae7(%rip)        # 84a0 <_ZL4MASK+0x1100>
    19b8:	00 
    19b9:	c4 e3 7d 02 d1 89    	vpblendd $0x89,%ymm1,%ymm0,%ymm2
    19bf:	c5 fd 7f 15 f9 6a 00 	vmovdqa %ymm2,0x6af9(%rip)        # 84c0 <_ZL4MASK+0x1120>
    19c6:	00 
    19c7:	c4 e3 7d 02 d1 8a    	vpblendd $0x8a,%ymm1,%ymm0,%ymm2
    19cd:	c5 fd 7f 15 0b 6b 00 	vmovdqa %ymm2,0x6b0b(%rip)        # 84e0 <_ZL4MASK+0x1140>
    19d4:	00 
    19d5:	c4 e3 7d 02 d1 8b    	vpblendd $0x8b,%ymm1,%ymm0,%ymm2
    19db:	c5 fd 7f 15 1d 6b 00 	vmovdqa %ymm2,0x6b1d(%rip)        # 8500 <_ZL4MASK+0x1160>
    19e2:	00 
    19e3:	c4 e3 7d 02 d1 8c    	vpblendd $0x8c,%ymm1,%ymm0,%ymm2
    19e9:	c5 fd 7f 15 2f 6b 00 	vmovdqa %ymm2,0x6b2f(%rip)        # 8520 <_ZL4MASK+0x1180>
    19f0:	00 
    19f1:	c4 e3 7d 02 d1 8d    	vpblendd $0x8d,%ymm1,%ymm0,%ymm2
    19f7:	c5 fd 7f 15 41 6b 00 	vmovdqa %ymm2,0x6b41(%rip)        # 8540 <_ZL4MASK+0x11a0>
    19fe:	00 
    19ff:	c4 e3 7d 02 d1 8e    	vpblendd $0x8e,%ymm1,%ymm0,%ymm2
    1a05:	c5 fd 7f 15 53 6b 00 	vmovdqa %ymm2,0x6b53(%rip)        # 8560 <_ZL4MASK+0x11c0>
    1a0c:	00 
    1a0d:	c4 e3 7d 02 d1 8f    	vpblendd $0x8f,%ymm1,%ymm0,%ymm2
    1a13:	c5 fd 7f 15 65 6b 00 	vmovdqa %ymm2,0x6b65(%rip)        # 8580 <_ZL4MASK+0x11e0>
    1a1a:	00 
    1a1b:	c4 e3 7d 02 d1 90    	vpblendd $0x90,%ymm1,%ymm0,%ymm2
    1a21:	c5 fd 7f 15 77 6b 00 	vmovdqa %ymm2,0x6b77(%rip)        # 85a0 <_ZL4MASK+0x1200>
    1a28:	00 
    1a29:	c4 e3 7d 02 d1 91    	vpblendd $0x91,%ymm1,%ymm0,%ymm2
    1a2f:	c5 fd 7f 15 89 6b 00 	vmovdqa %ymm2,0x6b89(%rip)        # 85c0 <_ZL4MASK+0x1220>
    1a36:	00 
    1a37:	c4 e3 7d 02 d1 92    	vpblendd $0x92,%ymm1,%ymm0,%ymm2
    1a3d:	c5 fd 7f 15 9b 6b 00 	vmovdqa %ymm2,0x6b9b(%rip)        # 85e0 <_ZL4MASK+0x1240>
    1a44:	00 
    1a45:	c4 e3 7d 02 d1 93    	vpblendd $0x93,%ymm1,%ymm0,%ymm2
    1a4b:	c5 fd 7f 15 ad 6b 00 	vmovdqa %ymm2,0x6bad(%rip)        # 8600 <_ZL4MASK+0x1260>
    1a52:	00 
    1a53:	c4 e3 7d 02 d1 94    	vpblendd $0x94,%ymm1,%ymm0,%ymm2
    1a59:	c5 fd 7f 15 bf 6b 00 	vmovdqa %ymm2,0x6bbf(%rip)        # 8620 <_ZL4MASK+0x1280>
    1a60:	00 
    1a61:	c4 e3 7d 02 d1 95    	vpblendd $0x95,%ymm1,%ymm0,%ymm2
    1a67:	c5 fd 7f 15 d1 6b 00 	vmovdqa %ymm2,0x6bd1(%rip)        # 8640 <_ZL4MASK+0x12a0>
    1a6e:	00 
    1a6f:	c4 e3 7d 02 d1 96    	vpblendd $0x96,%ymm1,%ymm0,%ymm2
    1a75:	c5 fd 7f 15 e3 6b 00 	vmovdqa %ymm2,0x6be3(%rip)        # 8660 <_ZL4MASK+0x12c0>
    1a7c:	00 
    1a7d:	c4 e3 7d 02 d1 97    	vpblendd $0x97,%ymm1,%ymm0,%ymm2
    1a83:	c5 fd 7f 15 f5 6b 00 	vmovdqa %ymm2,0x6bf5(%rip)        # 8680 <_ZL4MASK+0x12e0>
    1a8a:	00 
    1a8b:	c4 e3 7d 02 d1 98    	vpblendd $0x98,%ymm1,%ymm0,%ymm2
    1a91:	c5 fd 7f 15 07 6c 00 	vmovdqa %ymm2,0x6c07(%rip)        # 86a0 <_ZL4MASK+0x1300>
    1a98:	00 
    1a99:	c4 e3 7d 02 d1 99    	vpblendd $0x99,%ymm1,%ymm0,%ymm2
    1a9f:	c5 fd 7f 15 19 6c 00 	vmovdqa %ymm2,0x6c19(%rip)        # 86c0 <_ZL4MASK+0x1320>
    1aa6:	00 
    1aa7:	c4 e3 7d 02 d1 9a    	vpblendd $0x9a,%ymm1,%ymm0,%ymm2
    1aad:	c5 fd 7f 15 2b 6c 00 	vmovdqa %ymm2,0x6c2b(%rip)        # 86e0 <_ZL4MASK+0x1340>
    1ab4:	00 
    1ab5:	c4 e3 7d 02 d1 9b    	vpblendd $0x9b,%ymm1,%ymm0,%ymm2
    1abb:	c5 fd 7f 15 3d 6c 00 	vmovdqa %ymm2,0x6c3d(%rip)        # 8700 <_ZL4MASK+0x1360>
    1ac2:	00 
    1ac3:	c4 e3 7d 02 d1 9c    	vpblendd $0x9c,%ymm1,%ymm0,%ymm2
    1ac9:	c5 fd 7f 15 4f 6c 00 	vmovdqa %ymm2,0x6c4f(%rip)        # 8720 <_ZL4MASK+0x1380>
    1ad0:	00 
    1ad1:	c4 e3 7d 02 d1 9d    	vpblendd $0x9d,%ymm1,%ymm0,%ymm2
    1ad7:	c5 fd 7f 15 61 6c 00 	vmovdqa %ymm2,0x6c61(%rip)        # 8740 <_ZL4MASK+0x13a0>
    1ade:	00 
    1adf:	c4 e3 7d 02 d1 9e    	vpblendd $0x9e,%ymm1,%ymm0,%ymm2
    1ae5:	c5 fd 7f 15 73 6c 00 	vmovdqa %ymm2,0x6c73(%rip)        # 8760 <_ZL4MASK+0x13c0>
    1aec:	00 
    1aed:	c4 e3 7d 02 d1 9f    	vpblendd $0x9f,%ymm1,%ymm0,%ymm2
    1af3:	c5 fd 7f 15 85 6c 00 	vmovdqa %ymm2,0x6c85(%rip)        # 8780 <_ZL4MASK+0x13e0>
    1afa:	00 
    1afb:	c4 e3 7d 02 d1 a0    	vpblendd $0xa0,%ymm1,%ymm0,%ymm2
    1b01:	c5 fd 7f 15 97 6c 00 	vmovdqa %ymm2,0x6c97(%rip)        # 87a0 <_ZL4MASK+0x1400>
    1b08:	00 
    1b09:	c4 e3 7d 02 d1 a1    	vpblendd $0xa1,%ymm1,%ymm0,%ymm2
    1b0f:	c5 fd 7f 15 a9 6c 00 	vmovdqa %ymm2,0x6ca9(%rip)        # 87c0 <_ZL4MASK+0x1420>
    1b16:	00 
    1b17:	c4 e3 7d 02 d1 a2    	vpblendd $0xa2,%ymm1,%ymm0,%ymm2
    1b1d:	c5 fd 7f 15 bb 6c 00 	vmovdqa %ymm2,0x6cbb(%rip)        # 87e0 <_ZL4MASK+0x1440>
    1b24:	00 
    1b25:	c4 e3 7d 02 d1 a3    	vpblendd $0xa3,%ymm1,%ymm0,%ymm2
    1b2b:	c5 fd 7f 15 cd 6c 00 	vmovdqa %ymm2,0x6ccd(%rip)        # 8800 <_ZL4MASK+0x1460>
    1b32:	00 
    1b33:	c4 e3 7d 02 d1 a4    	vpblendd $0xa4,%ymm1,%ymm0,%ymm2
    1b39:	c5 fd 7f 15 df 6c 00 	vmovdqa %ymm2,0x6cdf(%rip)        # 8820 <_ZL4MASK+0x1480>
    1b40:	00 
    1b41:	c4 e3 7d 02 d1 a5    	vpblendd $0xa5,%ymm1,%ymm0,%ymm2
    1b47:	c5 fd 7f 15 f1 6c 00 	vmovdqa %ymm2,0x6cf1(%rip)        # 8840 <_ZL4MASK+0x14a0>
    1b4e:	00 
    1b4f:	c4 e3 7d 02 d1 a6    	vpblendd $0xa6,%ymm1,%ymm0,%ymm2
    1b55:	c5 fd 7f 15 03 6d 00 	vmovdqa %ymm2,0x6d03(%rip)        # 8860 <_ZL4MASK+0x14c0>
    1b5c:	00 
    1b5d:	c4 e3 7d 02 d1 a7    	vpblendd $0xa7,%ymm1,%ymm0,%ymm2
    1b63:	c5 fd 7f 15 15 6d 00 	vmovdqa %ymm2,0x6d15(%rip)        # 8880 <_ZL4MASK+0x14e0>
    1b6a:	00 
    1b6b:	c4 e3 7d 02 d1 a8    	vpblendd $0xa8,%ymm1,%ymm0,%ymm2
    1b71:	c5 fd 7f 15 27 6d 00 	vmovdqa %ymm2,0x6d27(%rip)        # 88a0 <_ZL4MASK+0x1500>
    1b78:	00 
    1b79:	c4 e3 7d 02 d1 a9    	vpblendd $0xa9,%ymm1,%ymm0,%ymm2
    1b7f:	c5 fd 7f 15 39 6d 00 	vmovdqa %ymm2,0x6d39(%rip)        # 88c0 <_ZL4MASK+0x1520>
    1b86:	00 
    1b87:	c4 e3 7d 02 d1 aa    	vpblendd $0xaa,%ymm1,%ymm0,%ymm2
    1b8d:	c5 fd 7f 15 4b 6d 00 	vmovdqa %ymm2,0x6d4b(%rip)        # 88e0 <_ZL4MASK+0x1540>
    1b94:	00 
    1b95:	c4 e3 7d 02 d1 ab    	vpblendd $0xab,%ymm1,%ymm0,%ymm2
    1b9b:	c5 fd 7f 15 5d 6d 00 	vmovdqa %ymm2,0x6d5d(%rip)        # 8900 <_ZL4MASK+0x1560>
    1ba2:	00 
    1ba3:	c4 e3 7d 02 d1 ac    	vpblendd $0xac,%ymm1,%ymm0,%ymm2
    1ba9:	c5 fd 7f 15 6f 6d 00 	vmovdqa %ymm2,0x6d6f(%rip)        # 8920 <_ZL4MASK+0x1580>
    1bb0:	00 
    1bb1:	c4 e3 7d 02 d1 ad    	vpblendd $0xad,%ymm1,%ymm0,%ymm2
    1bb7:	c5 fd 7f 15 81 6d 00 	vmovdqa %ymm2,0x6d81(%rip)        # 8940 <_ZL4MASK+0x15a0>
    1bbe:	00 
    1bbf:	c4 e3 7d 02 d1 ae    	vpblendd $0xae,%ymm1,%ymm0,%ymm2
    1bc5:	c5 fd 7f 15 93 6d 00 	vmovdqa %ymm2,0x6d93(%rip)        # 8960 <_ZL4MASK+0x15c0>
    1bcc:	00 
    1bcd:	c4 e3 7d 02 d1 af    	vpblendd $0xaf,%ymm1,%ymm0,%ymm2
    1bd3:	c5 fd 7f 15 a5 6d 00 	vmovdqa %ymm2,0x6da5(%rip)        # 8980 <_ZL4MASK+0x15e0>
    1bda:	00 
    1bdb:	c4 e3 7d 02 d1 b0    	vpblendd $0xb0,%ymm1,%ymm0,%ymm2
    1be1:	c5 fd 7f 15 b7 6d 00 	vmovdqa %ymm2,0x6db7(%rip)        # 89a0 <_ZL4MASK+0x1600>
    1be8:	00 
    1be9:	c4 e3 7d 02 d1 b1    	vpblendd $0xb1,%ymm1,%ymm0,%ymm2
    1bef:	c5 fd 7f 15 c9 6d 00 	vmovdqa %ymm2,0x6dc9(%rip)        # 89c0 <_ZL4MASK+0x1620>
    1bf6:	00 
    1bf7:	c4 e3 7d 02 d1 b2    	vpblendd $0xb2,%ymm1,%ymm0,%ymm2
    1bfd:	c5 fd 7f 15 db 6d 00 	vmovdqa %ymm2,0x6ddb(%rip)        # 89e0 <_ZL4MASK+0x1640>
    1c04:	00 
    1c05:	c4 e3 7d 02 d1 b3    	vpblendd $0xb3,%ymm1,%ymm0,%ymm2
    1c0b:	c5 fd 7f 15 ed 6d 00 	vmovdqa %ymm2,0x6ded(%rip)        # 8a00 <_ZL4MASK+0x1660>
    1c12:	00 
    1c13:	c4 e3 7d 02 d1 b4    	vpblendd $0xb4,%ymm1,%ymm0,%ymm2
    1c19:	c5 fd 7f 15 ff 6d 00 	vmovdqa %ymm2,0x6dff(%rip)        # 8a20 <_ZL4MASK+0x1680>
    1c20:	00 
    1c21:	c4 e3 7d 02 d1 b5    	vpblendd $0xb5,%ymm1,%ymm0,%ymm2
    1c27:	c5 fd 7f 15 11 6e 00 	vmovdqa %ymm2,0x6e11(%rip)        # 8a40 <_ZL4MASK+0x16a0>
    1c2e:	00 
    1c2f:	c4 e3 7d 02 d1 b6    	vpblendd $0xb6,%ymm1,%ymm0,%ymm2
    1c35:	c5 fd 7f 15 23 6e 00 	vmovdqa %ymm2,0x6e23(%rip)        # 8a60 <_ZL4MASK+0x16c0>
    1c3c:	00 
    1c3d:	c4 e3 7d 02 d1 b7    	vpblendd $0xb7,%ymm1,%ymm0,%ymm2
    1c43:	c5 fd 7f 15 35 6e 00 	vmovdqa %ymm2,0x6e35(%rip)        # 8a80 <_ZL4MASK+0x16e0>
    1c4a:	00 
    1c4b:	c4 e3 7d 02 d1 b8    	vpblendd $0xb8,%ymm1,%ymm0,%ymm2
    1c51:	c5 fd 7f 15 47 6e 00 	vmovdqa %ymm2,0x6e47(%rip)        # 8aa0 <_ZL4MASK+0x1700>
    1c58:	00 
    1c59:	c4 e3 7d 02 d1 b9    	vpblendd $0xb9,%ymm1,%ymm0,%ymm2
    1c5f:	c5 fd 7f 15 59 6e 00 	vmovdqa %ymm2,0x6e59(%rip)        # 8ac0 <_ZL4MASK+0x1720>
    1c66:	00 
    1c67:	c4 e3 7d 02 d1 ba    	vpblendd $0xba,%ymm1,%ymm0,%ymm2
    1c6d:	c5 fd 7f 15 6b 6e 00 	vmovdqa %ymm2,0x6e6b(%rip)        # 8ae0 <_ZL4MASK+0x1740>
    1c74:	00 
    1c75:	c4 e3 7d 02 d1 bb    	vpblendd $0xbb,%ymm1,%ymm0,%ymm2
    1c7b:	c5 fd 7f 15 7d 6e 00 	vmovdqa %ymm2,0x6e7d(%rip)        # 8b00 <_ZL4MASK+0x1760>
    1c82:	00 
    1c83:	c4 e3 7d 02 d1 bc    	vpblendd $0xbc,%ymm1,%ymm0,%ymm2
    1c89:	c5 fd 7f 15 8f 6e 00 	vmovdqa %ymm2,0x6e8f(%rip)        # 8b20 <_ZL4MASK+0x1780>
    1c90:	00 
    1c91:	c4 e3 7d 02 d1 bd    	vpblendd $0xbd,%ymm1,%ymm0,%ymm2
    1c97:	c5 fd 7f 15 a1 6e 00 	vmovdqa %ymm2,0x6ea1(%rip)        # 8b40 <_ZL4MASK+0x17a0>
    1c9e:	00 
    1c9f:	c4 e3 7d 02 d1 be    	vpblendd $0xbe,%ymm1,%ymm0,%ymm2
    1ca5:	c5 fd 7f 15 b3 6e 00 	vmovdqa %ymm2,0x6eb3(%rip)        # 8b60 <_ZL4MASK+0x17c0>
    1cac:	00 
    1cad:	c4 e3 7d 02 d1 bf    	vpblendd $0xbf,%ymm1,%ymm0,%ymm2
    1cb3:	c5 fd 7f 15 c5 6e 00 	vmovdqa %ymm2,0x6ec5(%rip)        # 8b80 <_ZL4MASK+0x17e0>
    1cba:	00 
    1cbb:	c4 e3 7d 02 d1 c0    	vpblendd $0xc0,%ymm1,%ymm0,%ymm2
    1cc1:	c5 fd 7f 15 d7 6e 00 	vmovdqa %ymm2,0x6ed7(%rip)        # 8ba0 <_ZL4MASK+0x1800>
    1cc8:	00 
    1cc9:	c4 e3 7d 02 d1 c1    	vpblendd $0xc1,%ymm1,%ymm0,%ymm2
    1ccf:	c5 fd 7f 15 e9 6e 00 	vmovdqa %ymm2,0x6ee9(%rip)        # 8bc0 <_ZL4MASK+0x1820>
    1cd6:	00 
    1cd7:	c4 e3 7d 02 d1 c2    	vpblendd $0xc2,%ymm1,%ymm0,%ymm2
    1cdd:	c5 fd 7f 15 fb 6e 00 	vmovdqa %ymm2,0x6efb(%rip)        # 8be0 <_ZL4MASK+0x1840>
    1ce4:	00 
    1ce5:	c4 e3 7d 02 d1 c3    	vpblendd $0xc3,%ymm1,%ymm0,%ymm2
    1ceb:	c5 fd 7f 15 0d 6f 00 	vmovdqa %ymm2,0x6f0d(%rip)        # 8c00 <_ZL4MASK+0x1860>
    1cf2:	00 
    1cf3:	c4 e3 7d 02 d1 c4    	vpblendd $0xc4,%ymm1,%ymm0,%ymm2
    1cf9:	c5 fd 7f 15 1f 6f 00 	vmovdqa %ymm2,0x6f1f(%rip)        # 8c20 <_ZL4MASK+0x1880>
    1d00:	00 
    1d01:	c4 e3 7d 02 d1 c5    	vpblendd $0xc5,%ymm1,%ymm0,%ymm2
    1d07:	c5 fd 7f 15 31 6f 00 	vmovdqa %ymm2,0x6f31(%rip)        # 8c40 <_ZL4MASK+0x18a0>
    1d0e:	00 
    1d0f:	c4 e3 7d 02 d1 c6    	vpblendd $0xc6,%ymm1,%ymm0,%ymm2
    1d15:	c5 fd 7f 15 43 6f 00 	vmovdqa %ymm2,0x6f43(%rip)        # 8c60 <_ZL4MASK+0x18c0>
    1d1c:	00 
    1d1d:	c4 e3 7d 02 d1 c7    	vpblendd $0xc7,%ymm1,%ymm0,%ymm2
    1d23:	c5 fd 7f 15 55 6f 00 	vmovdqa %ymm2,0x6f55(%rip)        # 8c80 <_ZL4MASK+0x18e0>
    1d2a:	00 
    1d2b:	c4 e3 7d 02 d1 c8    	vpblendd $0xc8,%ymm1,%ymm0,%ymm2
    1d31:	c5 fd 7f 15 67 6f 00 	vmovdqa %ymm2,0x6f67(%rip)        # 8ca0 <_ZL4MASK+0x1900>
    1d38:	00 
    1d39:	c4 e3 7d 02 d1 c9    	vpblendd $0xc9,%ymm1,%ymm0,%ymm2
    1d3f:	c5 fd 7f 15 79 6f 00 	vmovdqa %ymm2,0x6f79(%rip)        # 8cc0 <_ZL4MASK+0x1920>
    1d46:	00 
    1d47:	c4 e3 7d 02 d1 ca    	vpblendd $0xca,%ymm1,%ymm0,%ymm2
    1d4d:	c5 fd 7f 15 8b 6f 00 	vmovdqa %ymm2,0x6f8b(%rip)        # 8ce0 <_ZL4MASK+0x1940>
    1d54:	00 
    1d55:	c4 e3 7d 02 d1 cb    	vpblendd $0xcb,%ymm1,%ymm0,%ymm2
    1d5b:	c5 fd 7f 15 9d 6f 00 	vmovdqa %ymm2,0x6f9d(%rip)        # 8d00 <_ZL4MASK+0x1960>
    1d62:	00 
    1d63:	c4 e3 7d 02 d1 cc    	vpblendd $0xcc,%ymm1,%ymm0,%ymm2
    1d69:	c5 fd 7f 15 af 6f 00 	vmovdqa %ymm2,0x6faf(%rip)        # 8d20 <_ZL4MASK+0x1980>
    1d70:	00 
    1d71:	c4 e3 7d 02 d1 cd    	vpblendd $0xcd,%ymm1,%ymm0,%ymm2
    1d77:	c5 fd 7f 15 c1 6f 00 	vmovdqa %ymm2,0x6fc1(%rip)        # 8d40 <_ZL4MASK+0x19a0>
    1d7e:	00 
    1d7f:	c4 e3 7d 02 d1 ce    	vpblendd $0xce,%ymm1,%ymm0,%ymm2
    1d85:	c5 fd 7f 15 d3 6f 00 	vmovdqa %ymm2,0x6fd3(%rip)        # 8d60 <_ZL4MASK+0x19c0>
    1d8c:	00 
    1d8d:	c4 e3 7d 02 d1 cf    	vpblendd $0xcf,%ymm1,%ymm0,%ymm2
    1d93:	c5 fd 7f 15 e5 6f 00 	vmovdqa %ymm2,0x6fe5(%rip)        # 8d80 <_ZL4MASK+0x19e0>
    1d9a:	00 
    1d9b:	c4 e3 7d 02 d1 d0    	vpblendd $0xd0,%ymm1,%ymm0,%ymm2
    1da1:	c5 fd 7f 15 f7 6f 00 	vmovdqa %ymm2,0x6ff7(%rip)        # 8da0 <_ZL4MASK+0x1a00>
    1da8:	00 
    1da9:	c4 e3 7d 02 d1 d1    	vpblendd $0xd1,%ymm1,%ymm0,%ymm2
    1daf:	c5 fd 7f 15 09 70 00 	vmovdqa %ymm2,0x7009(%rip)        # 8dc0 <_ZL4MASK+0x1a20>
    1db6:	00 
    1db7:	c4 e3 7d 02 d1 d2    	vpblendd $0xd2,%ymm1,%ymm0,%ymm2
    1dbd:	c5 fd 7f 15 1b 70 00 	vmovdqa %ymm2,0x701b(%rip)        # 8de0 <_ZL4MASK+0x1a40>
    1dc4:	00 
    1dc5:	c4 e3 7d 02 d1 d3    	vpblendd $0xd3,%ymm1,%ymm0,%ymm2
    1dcb:	c5 fd 7f 15 2d 70 00 	vmovdqa %ymm2,0x702d(%rip)        # 8e00 <_ZL4MASK+0x1a60>
    1dd2:	00 
    1dd3:	c4 e3 7d 02 d1 d4    	vpblendd $0xd4,%ymm1,%ymm0,%ymm2
    1dd9:	c5 fd 7f 15 3f 70 00 	vmovdqa %ymm2,0x703f(%rip)        # 8e20 <_ZL4MASK+0x1a80>
    1de0:	00 
    1de1:	c4 e3 7d 02 d1 d5    	vpblendd $0xd5,%ymm1,%ymm0,%ymm2
    1de7:	c5 fd 7f 15 51 70 00 	vmovdqa %ymm2,0x7051(%rip)        # 8e40 <_ZL4MASK+0x1aa0>
    1dee:	00 
    1def:	c4 e3 7d 02 d1 d6    	vpblendd $0xd6,%ymm1,%ymm0,%ymm2
    1df5:	c5 fd 7f 15 63 70 00 	vmovdqa %ymm2,0x7063(%rip)        # 8e60 <_ZL4MASK+0x1ac0>
    1dfc:	00 
    1dfd:	c4 e3 7d 02 d1 d7    	vpblendd $0xd7,%ymm1,%ymm0,%ymm2
    1e03:	c5 fd 7f 15 75 70 00 	vmovdqa %ymm2,0x7075(%rip)        # 8e80 <_ZL4MASK+0x1ae0>
    1e0a:	00 
    1e0b:	c4 e3 7d 02 d1 d8    	vpblendd $0xd8,%ymm1,%ymm0,%ymm2
    1e11:	c5 fd 7f 15 87 70 00 	vmovdqa %ymm2,0x7087(%rip)        # 8ea0 <_ZL4MASK+0x1b00>
    1e18:	00 
    1e19:	c4 e3 7d 02 d1 d9    	vpblendd $0xd9,%ymm1,%ymm0,%ymm2
    1e1f:	c5 fd 7f 15 99 70 00 	vmovdqa %ymm2,0x7099(%rip)        # 8ec0 <_ZL4MASK+0x1b20>
    1e26:	00 
    1e27:	c4 e3 7d 02 d1 da    	vpblendd $0xda,%ymm1,%ymm0,%ymm2
    1e2d:	c5 fd 7f 15 ab 70 00 	vmovdqa %ymm2,0x70ab(%rip)        # 8ee0 <_ZL4MASK+0x1b40>
    1e34:	00 
    1e35:	c4 e3 7d 02 d1 db    	vpblendd $0xdb,%ymm1,%ymm0,%ymm2
    1e3b:	c5 fd 7f 15 bd 70 00 	vmovdqa %ymm2,0x70bd(%rip)        # 8f00 <_ZL4MASK+0x1b60>
    1e42:	00 
    1e43:	c4 e3 7d 02 d1 dc    	vpblendd $0xdc,%ymm1,%ymm0,%ymm2
    1e49:	c5 fd 7f 15 cf 70 00 	vmovdqa %ymm2,0x70cf(%rip)        # 8f20 <_ZL4MASK+0x1b80>
    1e50:	00 
    1e51:	c4 e3 7d 02 d1 dd    	vpblendd $0xdd,%ymm1,%ymm0,%ymm2
    1e57:	c5 fd 7f 15 e1 70 00 	vmovdqa %ymm2,0x70e1(%rip)        # 8f40 <_ZL4MASK+0x1ba0>
    1e5e:	00 
    1e5f:	c4 e3 7d 02 d1 de    	vpblendd $0xde,%ymm1,%ymm0,%ymm2
    1e65:	c5 fd 7f 15 f3 70 00 	vmovdqa %ymm2,0x70f3(%rip)        # 8f60 <_ZL4MASK+0x1bc0>
    1e6c:	00 
    1e6d:	c4 e3 7d 02 d1 df    	vpblendd $0xdf,%ymm1,%ymm0,%ymm2
    1e73:	c5 fd 7f 15 05 71 00 	vmovdqa %ymm2,0x7105(%rip)        # 8f80 <_ZL4MASK+0x1be0>
    1e7a:	00 
    1e7b:	c4 e3 7d 02 d1 e0    	vpblendd $0xe0,%ymm1,%ymm0,%ymm2
    1e81:	c5 fd 7f 15 17 71 00 	vmovdqa %ymm2,0x7117(%rip)        # 8fa0 <_ZL4MASK+0x1c00>
    1e88:	00 
    1e89:	c4 e3 7d 02 d1 e1    	vpblendd $0xe1,%ymm1,%ymm0,%ymm2
    1e8f:	c5 fd 7f 15 29 71 00 	vmovdqa %ymm2,0x7129(%rip)        # 8fc0 <_ZL4MASK+0x1c20>
    1e96:	00 
    1e97:	c4 e3 7d 02 d1 e2    	vpblendd $0xe2,%ymm1,%ymm0,%ymm2
    1e9d:	c5 fd 7f 15 3b 71 00 	vmovdqa %ymm2,0x713b(%rip)        # 8fe0 <_ZL4MASK+0x1c40>
    1ea4:	00 
    1ea5:	c4 e3 7d 02 d1 e3    	vpblendd $0xe3,%ymm1,%ymm0,%ymm2
    1eab:	c5 fd 7f 15 4d 71 00 	vmovdqa %ymm2,0x714d(%rip)        # 9000 <_ZL4MASK+0x1c60>
    1eb2:	00 
    1eb3:	c4 e3 7d 02 d1 e4    	vpblendd $0xe4,%ymm1,%ymm0,%ymm2
    1eb9:	c5 fd 7f 15 5f 71 00 	vmovdqa %ymm2,0x715f(%rip)        # 9020 <_ZL4MASK+0x1c80>
    1ec0:	00 
    1ec1:	c4 e3 7d 02 d1 e5    	vpblendd $0xe5,%ymm1,%ymm0,%ymm2
    1ec7:	c5 fd 7f 15 71 71 00 	vmovdqa %ymm2,0x7171(%rip)        # 9040 <_ZL4MASK+0x1ca0>
    1ece:	00 
    1ecf:	c4 e3 7d 02 d1 e6    	vpblendd $0xe6,%ymm1,%ymm0,%ymm2
    1ed5:	c5 fd 7f 15 83 71 00 	vmovdqa %ymm2,0x7183(%rip)        # 9060 <_ZL4MASK+0x1cc0>
    1edc:	00 
    1edd:	c4 e3 7d 02 d1 e7    	vpblendd $0xe7,%ymm1,%ymm0,%ymm2
    1ee3:	c5 fd 7f 15 95 71 00 	vmovdqa %ymm2,0x7195(%rip)        # 9080 <_ZL4MASK+0x1ce0>
    1eea:	00 
    1eeb:	c4 e3 7d 02 d1 e8    	vpblendd $0xe8,%ymm1,%ymm0,%ymm2
    1ef1:	c5 fd 7f 15 a7 71 00 	vmovdqa %ymm2,0x71a7(%rip)        # 90a0 <_ZL4MASK+0x1d00>
    1ef8:	00 
    1ef9:	c4 e3 7d 02 d1 e9    	vpblendd $0xe9,%ymm1,%ymm0,%ymm2
    1eff:	c5 fd 7f 15 b9 71 00 	vmovdqa %ymm2,0x71b9(%rip)        # 90c0 <_ZL4MASK+0x1d20>
    1f06:	00 
    1f07:	c4 e3 7d 02 d1 ea    	vpblendd $0xea,%ymm1,%ymm0,%ymm2
    1f0d:	c5 fd 7f 15 cb 71 00 	vmovdqa %ymm2,0x71cb(%rip)        # 90e0 <_ZL4MASK+0x1d40>
    1f14:	00 
    1f15:	c4 e3 7d 02 d1 eb    	vpblendd $0xeb,%ymm1,%ymm0,%ymm2
    1f1b:	c5 fd 7f 15 dd 71 00 	vmovdqa %ymm2,0x71dd(%rip)        # 9100 <_ZL4MASK+0x1d60>
    1f22:	00 
    1f23:	c4 e3 7d 02 d1 ec    	vpblendd $0xec,%ymm1,%ymm0,%ymm2
    1f29:	c5 fd 7f 15 ef 71 00 	vmovdqa %ymm2,0x71ef(%rip)        # 9120 <_ZL4MASK+0x1d80>
    1f30:	00 
    1f31:	c4 e3 7d 02 d1 ed    	vpblendd $0xed,%ymm1,%ymm0,%ymm2
    1f37:	c5 fd 7f 15 01 72 00 	vmovdqa %ymm2,0x7201(%rip)        # 9140 <_ZL4MASK+0x1da0>
    1f3e:	00 
    1f3f:	c4 e3 7d 02 d1 ee    	vpblendd $0xee,%ymm1,%ymm0,%ymm2
    1f45:	c5 fd 7f 15 13 72 00 	vmovdqa %ymm2,0x7213(%rip)        # 9160 <_ZL4MASK+0x1dc0>
    1f4c:	00 
    1f4d:	c4 e3 7d 02 d1 ef    	vpblendd $0xef,%ymm1,%ymm0,%ymm2
    1f53:	c5 fd 7f 15 25 72 00 	vmovdqa %ymm2,0x7225(%rip)        # 9180 <_ZL4MASK+0x1de0>
    1f5a:	00 
    1f5b:	c4 e3 7d 02 d1 f0    	vpblendd $0xf0,%ymm1,%ymm0,%ymm2
    1f61:	c5 fd 7f 15 37 72 00 	vmovdqa %ymm2,0x7237(%rip)        # 91a0 <_ZL4MASK+0x1e00>
    1f68:	00 
    1f69:	c4 e3 7d 02 d1 f1    	vpblendd $0xf1,%ymm1,%ymm0,%ymm2
    1f6f:	c5 fd 7f 15 49 72 00 	vmovdqa %ymm2,0x7249(%rip)        # 91c0 <_ZL4MASK+0x1e20>
    1f76:	00 
    1f77:	c4 e3 7d 02 d1 f2    	vpblendd $0xf2,%ymm1,%ymm0,%ymm2
    1f7d:	c5 fd 7f 15 5b 72 00 	vmovdqa %ymm2,0x725b(%rip)        # 91e0 <_ZL4MASK+0x1e40>
    1f84:	00 
    1f85:	c4 e3 7d 02 d1 f3    	vpblendd $0xf3,%ymm1,%ymm0,%ymm2
    1f8b:	c5 fd 7f 15 6d 72 00 	vmovdqa %ymm2,0x726d(%rip)        # 9200 <_ZL4MASK+0x1e60>
    1f92:	00 
    1f93:	c4 e3 7d 02 d1 f4    	vpblendd $0xf4,%ymm1,%ymm0,%ymm2
    1f99:	c5 fd 7f 15 7f 72 00 	vmovdqa %ymm2,0x727f(%rip)        # 9220 <_ZL4MASK+0x1e80>
    1fa0:	00 
    1fa1:	c4 e3 7d 02 d1 f5    	vpblendd $0xf5,%ymm1,%ymm0,%ymm2
    1fa7:	c5 fd 7f 15 91 72 00 	vmovdqa %ymm2,0x7291(%rip)        # 9240 <_ZL4MASK+0x1ea0>
    1fae:	00 
    1faf:	c4 e3 7d 02 d1 f6    	vpblendd $0xf6,%ymm1,%ymm0,%ymm2
    1fb5:	c5 fd 7f 15 a3 72 00 	vmovdqa %ymm2,0x72a3(%rip)        # 9260 <_ZL4MASK+0x1ec0>
    1fbc:	00 
    1fbd:	c4 e3 7d 02 d1 f7    	vpblendd $0xf7,%ymm1,%ymm0,%ymm2
    1fc3:	c5 fd 7f 15 b5 72 00 	vmovdqa %ymm2,0x72b5(%rip)        # 9280 <_ZL4MASK+0x1ee0>
    1fca:	00 
    1fcb:	c4 e3 7d 02 d1 f8    	vpblendd $0xf8,%ymm1,%ymm0,%ymm2
    1fd1:	c5 fd 7f 15 c7 72 00 	vmovdqa %ymm2,0x72c7(%rip)        # 92a0 <_ZL4MASK+0x1f00>
    1fd8:	00 
    1fd9:	c4 e3 7d 02 d1 f9    	vpblendd $0xf9,%ymm1,%ymm0,%ymm2
    1fdf:	c5 fd 7f 15 d9 72 00 	vmovdqa %ymm2,0x72d9(%rip)        # 92c0 <_ZL4MASK+0x1f20>
    1fe6:	00 
    1fe7:	c4 e3 7d 02 d1 fa    	vpblendd $0xfa,%ymm1,%ymm0,%ymm2
    1fed:	c5 fd 7f 15 eb 72 00 	vmovdqa %ymm2,0x72eb(%rip)        # 92e0 <_ZL4MASK+0x1f40>
    1ff4:	00 
    1ff5:	c4 e3 7d 02 d1 fb    	vpblendd $0xfb,%ymm1,%ymm0,%ymm2
    1ffb:	c5 fd 7f 15 fd 72 00 	vmovdqa %ymm2,0x72fd(%rip)        # 9300 <_ZL4MASK+0x1f60>
    2002:	00 
    2003:	c4 e3 7d 02 d1 fc    	vpblendd $0xfc,%ymm1,%ymm0,%ymm2
    2009:	c5 fd 7f 15 0f 73 00 	vmovdqa %ymm2,0x730f(%rip)        # 9320 <_ZL4MASK+0x1f80>
    2010:	00 
    2011:	c4 e3 7d 02 d1 fd    	vpblendd $0xfd,%ymm1,%ymm0,%ymm2
    2017:	c4 e3 7d 02 c1 fe    	vpblendd $0xfe,%ymm1,%ymm0,%ymm0
    201d:	c5 fd 7f 15 1b 73 00 	vmovdqa %ymm2,0x731b(%rip)        # 9340 <_ZL4MASK+0x1fa0>
    2024:	00 
    2025:	c5 fd 7f 05 33 73 00 	vmovdqa %ymm0,0x7333(%rip)        # 9360 <_ZL4MASK+0x1fc0>
    202c:	00 
    202d:	c5 fd 7f 0d 4b 73 00 	vmovdqa %ymm1,0x734b(%rip)        # 9380 <_ZL4MASK+0x1fe0>
    2034:	00 
    2035:	c5 f8 77             	vzeroupper 
    2038:	c9                   	leaveq 
    2039:	c3                   	retq   
    203a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000002040 <main>:
    2040:	55                   	push   %rbp
    2041:	31 ff                	xor    %edi,%edi
    2043:	48 89 e5             	mov    %rsp,%rbp
    2046:	41 57                	push   %r15
    2048:	41 56                	push   %r14
    204a:	41 55                	push   %r13
    204c:	41 54                	push   %r12
    204e:	53                   	push   %rbx
    204f:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    2053:	48 81 ec 00 15 00 00 	sub    $0x1500,%rsp
    205a:	e8 e1 ef ff ff       	callq  1040 <_ZNSt8ios_base15sync_with_stdioEb@plt>
    205f:	48 8d 5c 24 70       	lea    0x70(%rsp),%rbx
    2064:	ba 75 6c 00 00       	mov    $0x6c75,%edx
    2069:	c7 84 24 80 00 00 00 	movl   $0x61666564,0x80(%rsp)
    2070:	64 65 66 61 
    2074:	48 8d 43 10          	lea    0x10(%rbx),%rax
    2078:	48 89 de             	mov    %rbx,%rsi
    207b:	48 8d bc 24 70 01 00 	lea    0x170(%rsp),%rdi
    2082:	00 
    2083:	48 c7 05 7a 42 00 00 	movq   $0x0,0x427a(%rip)        # 6308 <_ZSt3cin@@GLIBCXX_3.4+0xe8>
    208a:	00 00 00 00 
    208e:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
    2093:	66 89 50 04          	mov    %dx,0x4(%rax)
    2097:	c6 40 06 74          	movb   $0x74,0x6(%rax)
    209b:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
    20a0:	48 c7 44 24 78 07 00 	movq   $0x7,0x78(%rsp)
    20a7:	00 00 
    20a9:	c6 84 24 87 00 00 00 	movb   $0x0,0x87(%rsp)
    20b0:	00 
    20b1:	48 89 7c 24 28       	mov    %rdi,0x28(%rsp)
    20b6:	e8 a5 f0 ff ff       	callq  1160 <_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@plt>
    20bb:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    20c0:	48 8d 43 10          	lea    0x10(%rbx),%rax
    20c4:	48 39 c7             	cmp    %rax,%rdi
    20c7:	74 05                	je     20ce <main+0x8e>
    20c9:	e8 d2 ef ff ff       	callq  10a0 <_ZdlPv@plt>
    20ce:	48 8b 5c 24 28       	mov    0x28(%rsp),%rbx
    20d3:	48 89 df             	mov    %rbx,%rdi
    20d6:	e8 25 f0 ff ff       	callq  1100 <_ZNSt13random_device9_M_getvalEv@plt>
    20db:	48 ba 05 00 00 00 02 	movabs $0x200000005,%rdx
    20e2:	00 00 00 
    20e5:	89 c1                	mov    %eax,%ecx
    20e7:	48 89 df             	mov    %rbx,%rdi
    20ea:	48 89 c8             	mov    %rcx,%rax
    20ed:	48 f7 e2             	mul    %rdx
    20f0:	48 89 c8             	mov    %rcx,%rax
    20f3:	48 29 d0             	sub    %rdx,%rax
    20f6:	48 d1 e8             	shr    %rax
    20f9:	48 01 d0             	add    %rdx,%rax
    20fc:	48 c1 e8 1e          	shr    $0x1e,%rax
    2100:	48 89 c2             	mov    %rax,%rdx
    2103:	48 c1 e2 1f          	shl    $0x1f,%rdx
    2107:	48 29 c2             	sub    %rax,%rdx
    210a:	48 29 d1             	sub    %rdx,%rcx
    210d:	ba 01 00 00 00       	mov    $0x1,%edx
    2112:	48 89 c8             	mov    %rcx,%rax
    2115:	48 0f 44 c2          	cmove  %rdx,%rax
    2119:	48 89 05 c0 72 00 00 	mov    %rax,0x72c0(%rip)        # 93e0 <_ZL9generator>
    2120:	e8 6b ef ff ff       	callq  1090 <_ZNSt13random_device7_M_finiEv@plt>
    2125:	be 00 00 40 00       	mov    $0x400000,%esi
    212a:	bf 80 00 00 00       	mov    $0x80,%edi
    212f:	e8 3c ef ff ff       	callq  1070 <aligned_alloc@plt>
    2134:	be 80 00 00 00       	mov    $0x80,%esi
    2139:	bf 80 00 00 00       	mov    $0x80,%edi
    213e:	49 89 c7             	mov    %rax,%r15
    2141:	e8 2a ef ff ff       	callq  1070 <aligned_alloc@plt>
    2146:	48 8d 35 0b 1f 00 00 	lea    0x1f0b(%rip),%rsi        # 4058 <_IO_stdin_used+0x58>
    214d:	48 8d 3d ac 3f 00 00 	lea    0x3fac(%rip),%rdi        # 6100 <_ZSt4cout@@GLIBCXX_3.4>
    2154:	49 89 c6             	mov    %rax,%r14
    2157:	e8 54 ef ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    215c:	c5 fd 6f 15 fc 1f 00 	vmovdqa 0x1ffc(%rip),%ymm2        # 4160 <_IO_stdin_used+0x160>
    2163:	00 
    2164:	4c 89 f8             	mov    %r15,%rax
    2167:	c5 fd 6f 2d 11 20 00 	vmovdqa 0x2011(%rip),%ymm5        # 4180 <_IO_stdin_used+0x180>
    216e:	00 
    216f:	c5 fd 6f 25 89 1f 00 	vmovdqa 0x1f89(%rip),%ymm4        # 4100 <_IO_stdin_used+0x100>
    2176:	00 
    2177:	c5 fd 28 1d 21 20 00 	vmovapd 0x2021(%rip),%ymm3        # 41a0 <_IO_stdin_used+0x1a0>
    217e:	00 
    217f:	49 8d 97 00 00 40 00 	lea    0x400000(%r15),%rdx
    2186:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    218d:	00 00 00 
    2190:	c5 ed fe c4          	vpaddd %ymm4,%ymm2,%ymm0
    2194:	c4 e3 7d 39 d6 01    	vextracti128 $0x1,%ymm2,%xmm6
    219a:	48 83 c0 20          	add    $0x20,%rax
    219e:	c5 fe e6 c8          	vcvtdq2pd %xmm0,%ymm1
    21a2:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
    21a8:	c5 fe e6 f6          	vcvtdq2pd %xmm6,%ymm6
    21ac:	c5 f5 59 cb          	vmulpd %ymm3,%ymm1,%ymm1
    21b0:	c5 fe e6 c0          	vcvtdq2pd %xmm0,%ymm0
    21b4:	c5 fd 59 c3          	vmulpd %ymm3,%ymm0,%ymm0
    21b8:	c5 cd 59 f3          	vmulpd %ymm3,%ymm6,%ymm6
    21bc:	c5 fd e6 c9          	vcvttpd2dq %ymm1,%xmm1
    21c0:	c5 fd e6 c0          	vcvttpd2dq %ymm0,%xmm0
    21c4:	c4 e3 75 38 c0 01    	vinserti128 $0x1,%xmm0,%ymm1,%ymm0
    21ca:	c5 fe e6 ca          	vcvtdq2pd %xmm2,%ymm1
    21ce:	c5 ed fe d5          	vpaddd %ymm5,%ymm2,%ymm2
    21d2:	c5 f5 59 cb          	vmulpd %ymm3,%ymm1,%ymm1
    21d6:	c5 fd e6 f6          	vcvttpd2dq %ymm6,%xmm6
    21da:	c5 fd e6 c9          	vcvttpd2dq %ymm1,%xmm1
    21de:	c4 e3 75 38 ce 01    	vinserti128 $0x1,%xmm6,%ymm1,%ymm1
    21e4:	c5 fd fa c1          	vpsubd %ymm1,%ymm0,%ymm0
    21e8:	c5 fe 7f 40 e0       	vmovdqu %ymm0,-0x20(%rax)
    21ed:	48 39 c2             	cmp    %rax,%rdx
    21f0:	75 9e                	jne    2190 <main+0x150>
    21f2:	48 8d 35 3d 1e 00 00 	lea    0x1e3d(%rip),%rsi        # 4036 <_IO_stdin_used+0x36>
    21f9:	48 8d 3d 00 3f 00 00 	lea    0x3f00(%rip),%rdi        # 6100 <_ZSt4cout@@GLIBCXX_3.4>
    2200:	c5 f8 77             	vzeroupper 
    2203:	e8 a8 ee ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    2208:	31 db                	xor    %ebx,%ebx
    220a:	45 31 e4             	xor    %r12d,%r12d
    220d:	48 8d 35 74 1e 00 00 	lea    0x1e74(%rip),%rsi        # 4088 <_IO_stdin_used+0x88>
    2214:	48 8d 3d e5 3e 00 00 	lea    0x3ee5(%rip),%rdi        # 6100 <_ZSt4cout@@GLIBCXX_3.4>
    221b:	e8 90 ee ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    2220:	31 c9                	xor    %ecx,%ecx
    2222:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
    2229:	00 00 
    222b:	eb 0f                	jmp    223c <main+0x1fc>
    222d:	0f 1f 00             	nopl   (%rax)
    2230:	48 ff c3             	inc    %rbx
    2233:	48 81 fb 00 00 10 00 	cmp    $0x100000,%rbx
    223a:	74 4b                	je     2287 <main+0x247>
    223c:	41 83 3c 9f 01       	cmpl   $0x1,(%r15,%rbx,4)
    2241:	75 ed                	jne    2230 <main+0x1f0>
    2243:	b8 ab aa aa aa       	mov    $0xaaaaaaab,%eax
    2248:	41 c7 04 9f 00 00 00 	movl   $0x0,(%r15,%rbx,4)
    224f:	00 
    2250:	f7 e3                	mul    %ebx
    2252:	89 d8                	mov    %ebx,%eax
    2254:	d1 ea                	shr    %edx
    2256:	29 d0                	sub    %edx,%eax
    2258:	83 e0 01             	and    $0x1,%eax
    225b:	44 8d ac 43 ff ff 0f 	lea    0xfffff(%rbx,%rax,2),%r13d
    2262:	00 
    2263:	41 81 e5 ff ff 0f 00 	and    $0xfffff,%r13d
    226a:	4c 39 e1             	cmp    %r12,%rcx
    226d:	0f 84 cd 01 00 00    	je     2440 <main+0x400>
    2273:	48 ff c3             	inc    %rbx
    2276:	45 89 2c 24          	mov    %r13d,(%r12)
    227a:	49 83 c4 04          	add    $0x4,%r12
    227e:	48 81 fb 00 00 10 00 	cmp    $0x100000,%rbx
    2285:	75 b5                	jne    223c <main+0x1fc>
    2287:	4c 2b 64 24 38       	sub    0x38(%rsp),%r12
    228c:	31 d2                	xor    %edx,%edx
    228e:	31 c0                	xor    %eax,%eax
    2290:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
    2295:	49 c1 fc 02          	sar    $0x2,%r12
    2299:	74 18                	je     22b3 <main+0x273>
    229b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    22a0:	48 63 14 91          	movslq (%rcx,%rdx,4),%rdx
    22a4:	41 ff 04 97          	incl   (%r15,%rdx,4)
    22a8:	8d 50 01             	lea    0x1(%rax),%edx
    22ab:	48 89 d0             	mov    %rdx,%rax
    22ae:	4c 39 e2             	cmp    %r12,%rdx
    22b1:	72 ed                	jb     22a0 <main+0x260>
    22b3:	48 83 7c 24 38 00    	cmpq   $0x0,0x38(%rsp)
    22b9:	74 0a                	je     22c5 <main+0x285>
    22bb:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    22c0:	e8 db ed ff ff       	callq  10a0 <_ZdlPv@plt>
    22c5:	48 8d 35 6a 1d 00 00 	lea    0x1d6a(%rip),%rsi        # 4036 <_IO_stdin_used+0x36>
    22cc:	48 8d 3d 2d 3e 00 00 	lea    0x3e2d(%rip),%rdi        # 6100 <_ZSt4cout@@GLIBCXX_3.4>
    22d3:	49 bc 05 00 00 00 02 	movabs $0x200000005,%r12
    22da:	00 00 00 
    22dd:	49 bd 21 00 00 00 04 	movabs $0x8000000400000021,%r13
    22e4:	00 00 80 
    22e7:	e8 c4 ed ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    22ec:	48 8d 35 cd 1d 00 00 	lea    0x1dcd(%rip),%rsi        # 40c0 <_IO_stdin_used+0xc0>
    22f3:	48 8d 3d 06 3e 00 00 	lea    0x3e06(%rip),%rdi        # 6100 <_ZSt4cout@@GLIBCXX_3.4>
    22fa:	e8 b1 ed ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    22ff:	48 8d 3d fa 3d 00 00 	lea    0x3dfa(%rip),%rdi        # 6100 <_ZSt4cout@@GLIBCXX_3.4>
    2306:	e8 55 ed ff ff       	callq  1060 <_ZNSo5flushEv@plt>
    230b:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    2310:	ba 10 00 00 00       	mov    $0x10,%edx
    2315:	48 8d 35 21 1d 00 00 	lea    0x1d21(%rip),%rsi        # 403d <_IO_stdin_used+0x3d>
    231c:	e8 af ed ff ff       	callq  10d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@plt>
    2321:	48 8d 44 24 4c       	lea    0x4c(%rsp),%rax
    2326:	c4 c1 f9 6e ef       	vmovq  %r15,%xmm5
    232b:	c7 44 24 38 00 00 00 	movl   $0x0,0x38(%rsp)
    2332:	00 
    2333:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    2338:	48 8d 44 24 50       	lea    0x50(%rsp),%rax
    233d:	c4 c3 d1 22 ee 01    	vpinsrq $0x1,%r14,%xmm5,%xmm5
    2343:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    2348:	48 8d 44 24 70       	lea    0x70(%rsp),%rax
    234d:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    2352:	c5 f8 29 2c 24       	vmovaps %xmm5,(%rsp)
    2357:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    235e:	00 00 
    2360:	b9 10 00 00 00       	mov    $0x10,%ecx
    2365:	31 c0                	xor    %eax,%eax
    2367:	4c 89 f7             	mov    %r14,%rdi
    236a:	31 f6                	xor    %esi,%esi
    236c:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    236f:	c7 44 24 4c 00 00 00 	movl   $0x0,0x4c(%rsp)
    2376:	00 
    2377:	eb 36                	jmp    23af <main+0x36f>
    2379:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2380:	48 83 fe 01          	cmp    $0x1,%rsi
    2384:	76 1c                	jbe    23a2 <main+0x362>
    2386:	41 8b 44 b6 fc       	mov    -0x4(%r14,%rsi,4),%eax
    238b:	41 03 44 b7 fc       	add    -0x4(%r15,%rsi,4),%eax
    2390:	83 f8 01             	cmp    $0x1,%eax
    2393:	41 89 44 b7 fc       	mov    %eax,-0x4(%r15,%rsi,4)
    2398:	0f 9f c0             	setg   %al
    239b:	0f b6 c0             	movzbl %al,%eax
    239e:	01 44 24 4c          	add    %eax,0x4c(%rsp)
    23a2:	48 ff c6             	inc    %rsi
    23a5:	48 83 fe 08          	cmp    $0x8,%rsi
    23a9:	0f 84 81 01 00 00    	je     2530 <main+0x4f0>
    23af:	45 8b 04 b7          	mov    (%r15,%rsi,4),%r8d
    23b3:	41 89 f1             	mov    %esi,%r9d
    23b6:	41 83 f8 01          	cmp    $0x1,%r8d
    23ba:	7e c4                	jle    2380 <main+0x340>
    23bc:	48 8b 0d 1d 70 00 00 	mov    0x701d(%rip),%rcx        # 93e0 <_ZL9generator>
    23c3:	4c 8d 15 16 70 00 00 	lea    0x7016(%rip),%r10        # 93e0 <_ZL9generator>
    23ca:	31 ff                	xor    %edi,%edi
    23cc:	0f 1f 40 00          	nopl   0x0(%rax)
    23d0:	4c 69 d9 a7 41 00 00 	imul   $0x41a7,%rcx,%r11
    23d7:	4c 89 d8             	mov    %r11,%rax
    23da:	4c 89 d9             	mov    %r11,%rcx
    23dd:	49 f7 e4             	mul    %r12
    23e0:	48 29 d1             	sub    %rdx,%rcx
    23e3:	48 d1 e9             	shr    %rcx
    23e6:	48 01 ca             	add    %rcx,%rdx
    23e9:	4c 89 d9             	mov    %r11,%rcx
    23ec:	48 c1 ea 1e          	shr    $0x1e,%rdx
    23f0:	48 89 d0             	mov    %rdx,%rax
    23f3:	48 c1 e0 1f          	shl    $0x1f,%rax
    23f7:	48 29 d0             	sub    %rdx,%rax
    23fa:	48 29 c1             	sub    %rax,%rcx
    23fd:	48 8d 41 ff          	lea    -0x1(%rcx),%rax
    2401:	48 3d fb ff ff 7f    	cmp    $0x7ffffffb,%rax
    2407:	77 c7                	ja     23d0 <main+0x390>
    2409:	48 d1 e8             	shr    %rax
    240c:	ff c7                	inc    %edi
    240e:	49 89 0a             	mov    %rcx,(%r10)
    2411:	49 f7 e5             	mul    %r13
    2414:	48 c1 ea 1c          	shr    $0x1c,%rdx
    2418:	41 8d 44 51 1f       	lea    0x1f(%r9,%rdx,2),%eax
    241d:	83 e0 1f             	and    $0x1f,%eax
    2420:	41 ff 04 86          	incl   (%r14,%rax,4)
    2424:	41 39 f8             	cmp    %edi,%r8d
    2427:	75 a7                	jne    23d0 <main+0x390>
    2429:	41 c7 04 b7 00 00 00 	movl   $0x0,(%r15,%rsi,4)
    2430:	00 
    2431:	e9 4a ff ff ff       	jmpq   2380 <main+0x340>
    2436:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    243d:	00 00 00 
    2440:	4c 89 e0             	mov    %r12,%rax
    2443:	48 2b 44 24 38       	sub    0x38(%rsp),%rax
    2448:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
    244d:	48 c1 f8 02          	sar    $0x2,%rax
    2451:	0f 84 c9 00 00 00    	je     2520 <main+0x4e0>
    2457:	48 c7 44 24 20 fc ff 	movq   $0xfffffffffffffffc,0x20(%rsp)
    245e:	ff ff 
    2460:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    2464:	48 39 d0             	cmp    %rdx,%rax
    2467:	76 77                	jbe    24e0 <main+0x4a0>
    2469:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    246e:	e8 4d ec ff ff       	callq  10c0 <_Znwm@plt>
    2473:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    2478:	49 89 c0             	mov    %rax,%r8
    247b:	48 01 c1             	add    %rax,%rcx
    247e:	48 8b 44 24 18       	mov    0x18(%rsp),%rax
    2483:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
    2488:	45 89 2c 00          	mov    %r13d,(%r8,%rax,1)
    248c:	4c 39 e6             	cmp    %r12,%rsi
    248f:	74 7f                	je     2510 <main+0x4d0>
    2491:	4c 89 c7             	mov    %r8,%rdi
    2494:	48 89 c2             	mov    %rax,%rdx
    2497:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    249c:	49 89 c4             	mov    %rax,%r12
    249f:	e8 ac ec ff ff       	callq  1150 <memmove@plt>
    24a4:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    24a9:	49 89 c0             	mov    %rax,%r8
    24ac:	4f 8d 64 20 04       	lea    0x4(%r8,%r12,1),%r12
    24b1:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    24b6:	4c 89 44 24 18       	mov    %r8,0x18(%rsp)
    24bb:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    24c0:	e8 db eb ff ff       	callq  10a0 <_ZdlPv@plt>
    24c5:	4c 8b 44 24 18       	mov    0x18(%rsp),%r8
    24ca:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    24cf:	4c 89 44 24 38       	mov    %r8,0x38(%rsp)
    24d4:	e9 57 fd ff ff       	jmpq   2230 <main+0x1f0>
    24d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    24e0:	48 bf ff ff ff ff ff 	movabs $0x3fffffffffffffff,%rdi
    24e7:	ff ff 3f 
    24ea:	48 39 fa             	cmp    %rdi,%rdx
    24ed:	0f 87 94 01 00 00    	ja     2687 <main+0x647>
    24f3:	48 85 d2             	test   %rdx,%rdx
    24f6:	0f 85 99 01 00 00    	jne    2695 <main+0x655>
    24fc:	31 c9                	xor    %ecx,%ecx
    24fe:	45 31 c0             	xor    %r8d,%r8d
    2501:	e9 78 ff ff ff       	jmpq   247e <main+0x43e>
    2506:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    250d:	00 00 00 
    2510:	48 83 7c 24 38 00    	cmpq   $0x0,0x38(%rsp)
    2516:	4d 8d 64 00 04       	lea    0x4(%r8,%rax,1),%r12
    251b:	74 b2                	je     24cf <main+0x48f>
    251d:	eb 92                	jmp    24b1 <main+0x471>
    251f:	90                   	nop
    2520:	48 c7 44 24 20 04 00 	movq   $0x4,0x20(%rsp)
    2527:	00 00 
    2529:	e9 3b ff ff ff       	jmpq   2469 <main+0x429>
    252e:	66 90                	xchg   %ax,%ax
    2530:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    2535:	31 c0                	xor    %eax,%eax
    2537:	b9 20 00 00 00       	mov    $0x20,%ecx
    253c:	31 d2                	xor    %edx,%edx
    253e:	c5 fa 7e 64 24 20    	vmovq  0x20(%rsp),%xmm4
    2544:	c5 f9 6f 3c 24       	vmovdqa (%rsp),%xmm7
    2549:	31 db                	xor    %ebx,%ebx
    254b:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    254e:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    2553:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
    2558:	48 8d 3d 71 08 00 00 	lea    0x871(%rip),%rdi        # 2dd0 <_Z9descargarPiS_._omp_fn.0>
    255f:	c5 f8 29 7c 24 50    	vmovaps %xmm7,0x50(%rsp)
    2565:	c4 e3 d9 22 c0 01    	vpinsrq $0x1,%rax,%xmm4,%xmm0
    256b:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    2570:	c5 f8 29 44 24 60    	vmovaps %xmm0,0x60(%rsp)
    2576:	e8 a5 eb ff ff       	callq  1120 <GOMP_parallel@plt>
    257b:	41 8b 77 1c          	mov    0x1c(%r15),%esi
    257f:	45 8b 47 20          	mov    0x20(%r15),%r8d
    2583:	41 8b 7e 20          	mov    0x20(%r14),%edi
    2587:	41 8b 4e 1c          	mov    0x1c(%r14),%ecx
    258b:	41 8b 56 7c          	mov    0x7c(%r14),%edx
    258f:	41 03 97 fc ff 3f 00 	add    0x3ffffc(%r15),%edx
    2596:	44 01 c7             	add    %r8d,%edi
    2599:	01 f1                	add    %esi,%ecx
    259b:	41 8b 06             	mov    (%r14),%eax
    259e:	41 03 07             	add    (%r15),%eax
    25a1:	83 fa 01             	cmp    $0x1,%edx
    25a4:	41 89 07             	mov    %eax,(%r15)
    25a7:	0f 9f c3             	setg   %bl
    25aa:	83 f8 01             	cmp    $0x1,%eax
    25ad:	41 89 7f 20          	mov    %edi,0x20(%r15)
    25b1:	0f 9f c0             	setg   %al
    25b4:	41 89 97 fc ff 3f 00 	mov    %edx,0x3ffffc(%r15)
    25bb:	0f b6 c0             	movzbl %al,%eax
    25be:	41 89 4f 1c          	mov    %ecx,0x1c(%r15)
    25c2:	01 c3                	add    %eax,%ebx
    25c4:	31 c0                	xor    %eax,%eax
    25c6:	03 5c 24 4c          	add    0x4c(%rsp),%ebx
    25ca:	41 83 f8 01          	cmp    $0x1,%r8d
    25ce:	0f 9f c0             	setg   %al
    25d1:	29 c3                	sub    %eax,%ebx
    25d3:	31 c0                	xor    %eax,%eax
    25d5:	83 ff 01             	cmp    $0x1,%edi
    25d8:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    25dd:	0f 9f c0             	setg   %al
    25e0:	01 c3                	add    %eax,%ebx
    25e2:	31 c0                	xor    %eax,%eax
    25e4:	83 fe 01             	cmp    $0x1,%esi
    25e7:	0f 9f c0             	setg   %al
    25ea:	29 c3                	sub    %eax,%ebx
    25ec:	31 c0                	xor    %eax,%eax
    25ee:	83 f9 01             	cmp    $0x1,%ecx
    25f1:	0f 9f c0             	setg   %al
    25f4:	01 c3                	add    %eax,%ebx
    25f6:	89 de                	mov    %ebx,%esi
    25f8:	e8 73 eb ff ff       	callq  1170 <_ZNSolsEi@plt>
    25fd:	ba 01 00 00 00       	mov    $0x1,%edx
    2602:	48 8d 35 0e 1a 00 00 	lea    0x1a0e(%rip),%rsi        # 4017 <_IO_stdin_used+0x17>
    2609:	48 89 c7             	mov    %rax,%rdi
    260c:	e8 cf ea ff ff       	callq  10e0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2611:	ff 44 24 38          	incl   0x38(%rsp)
    2615:	8b 44 24 38          	mov    0x38(%rsp),%eax
    2619:	85 db                	test   %ebx,%ebx
    261b:	7e 0b                	jle    2628 <main+0x5e8>
    261d:	3d 0f 27 00 00       	cmp    $0x270f,%eax
    2622:	0f 8e 38 fd ff ff    	jle    2360 <main+0x320>
    2628:	ba 07 00 00 00       	mov    $0x7,%edx
    262d:	48 8d 35 16 1a 00 00 	lea    0x1a16(%rip),%rsi        # 404a <_IO_stdin_used+0x4a>
    2634:	48 8d 3d c5 3a 00 00 	lea    0x3ac5(%rip),%rdi        # 6100 <_ZSt4cout@@GLIBCXX_3.4>
    263b:	e8 a0 ea ff ff       	callq  10e0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    2640:	85 db                	test   %ebx,%ebx
    2642:	48 8d 35 bb 19 00 00 	lea    0x19bb(%rip),%rsi        # 4004 <_IO_stdin_used+0x4>
    2649:	48 8d 05 c9 19 00 00 	lea    0x19c9(%rip),%rax        # 4019 <_IO_stdin_used+0x19>
    2650:	48 0f 4e f0          	cmovle %rax,%rsi
    2654:	48 8d 3d a5 3a 00 00 	lea    0x3aa5(%rip),%rdi        # 6100 <_ZSt4cout@@GLIBCXX_3.4>
    265b:	e8 50 ea ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    2660:	48 8d 3d 99 3a 00 00 	lea    0x3a99(%rip),%rdi        # 6100 <_ZSt4cout@@GLIBCXX_3.4>
    2667:	e8 f4 e9 ff ff       	callq  1060 <_ZNSo5flushEv@plt>
    266c:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    2671:	e8 9a ea ff ff       	callq  1110 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    2676:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
    267a:	31 c0                	xor    %eax,%eax
    267c:	5b                   	pop    %rbx
    267d:	41 5c                	pop    %r12
    267f:	41 5d                	pop    %r13
    2681:	41 5e                	pop    %r14
    2683:	41 5f                	pop    %r15
    2685:	5d                   	pop    %rbp
    2686:	c3                   	retq   
    2687:	48 c7 44 24 20 fc ff 	movq   $0xfffffffffffffffc,0x20(%rsp)
    268e:	ff ff 
    2690:	e9 d4 fd ff ff       	jmpq   2469 <main+0x429>
    2695:	48 c1 e0 03          	shl    $0x3,%rax
    2699:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    269e:	e9 c6 fd ff ff       	jmpq   2469 <main+0x429>
    26a3:	48 89 c3             	mov    %rax,%rbx
    26a6:	eb 05                	jmp    26ad <main+0x66d>
    26a8:	48 89 c3             	mov    %rax,%rbx
    26ab:	eb 17                	jmp    26c4 <main+0x684>
    26ad:	48 83 7c 24 38 00    	cmpq   $0x0,0x38(%rsp)
    26b3:	74 37                	je     26ec <main+0x6ac>
    26b5:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    26ba:	c5 f8 77             	vzeroupper 
    26bd:	e8 de e9 ff ff       	callq  10a0 <_ZdlPv@plt>
    26c2:	eb 0d                	jmp    26d1 <main+0x691>
    26c4:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    26c9:	c5 f8 77             	vzeroupper 
    26cc:	e8 bf e9 ff ff       	callq  1090 <_ZNSt13random_device7_M_finiEv@plt>
    26d1:	48 89 df             	mov    %rbx,%rdi
    26d4:	e8 a7 ea ff ff       	callq  1180 <_Unwind_Resume@plt>
    26d9:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
    26de:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    26e3:	48 83 c0 10          	add    $0x10,%rax
    26e7:	48 39 c7             	cmp    %rax,%rdi
    26ea:	75 ce                	jne    26ba <main+0x67a>
    26ec:	c5 f8 77             	vzeroupper 
    26ef:	eb e0                	jmp    26d1 <main+0x691>
    26f1:	48 89 c3             	mov    %rax,%rbx
    26f4:	eb e3                	jmp    26d9 <main+0x699>
    26f6:	48 89 c3             	mov    %rax,%rbx
    26f9:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    26fe:	c5 f8 77             	vzeroupper 
    2701:	e8 0a ea ff ff       	callq  1110 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    2706:	eb c9                	jmp    26d1 <main+0x691>
    2708:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    270f:	00 

0000000000002710 <set_fast_math>:
    2710:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
    2715:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
    271c:	00 
    271d:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
    2722:	c3                   	retq   
    2723:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    272a:	00 00 00 
    272d:	0f 1f 00             	nopl   (%rax)

0000000000002730 <_start>:
    2730:	31 ed                	xor    %ebp,%ebp
    2732:	49 89 d1             	mov    %rdx,%r9
    2735:	5e                   	pop    %rsi
    2736:	48 89 e2             	mov    %rsp,%rdx
    2739:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    273d:	50                   	push   %rax
    273e:	54                   	push   %rsp
    273f:	4c 8d 05 2a 0a 00 00 	lea    0xa2a(%rip),%r8        # 3170 <__libc_csu_fini>
    2746:	48 8d 0d b3 09 00 00 	lea    0x9b3(%rip),%rcx        # 3100 <__libc_csu_init>
    274d:	48 8d 3d ec f8 ff ff 	lea    -0x714(%rip),%rdi        # 2040 <main>
    2754:	ff 15 86 38 00 00    	callq  *0x3886(%rip)        # 5fe0 <__libc_start_main@GLIBC_2.2.5>
    275a:	f4                   	hlt    
    275b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000002760 <deregister_tm_clones>:
    2760:	48 8d 3d 89 39 00 00 	lea    0x3989(%rip),%rdi        # 60f0 <__TMC_END__>
    2767:	48 8d 05 82 39 00 00 	lea    0x3982(%rip),%rax        # 60f0 <__TMC_END__>
    276e:	48 39 f8             	cmp    %rdi,%rax
    2771:	74 15                	je     2788 <deregister_tm_clones+0x28>
    2773:	48 8b 05 5e 38 00 00 	mov    0x385e(%rip),%rax        # 5fd8 <_ITM_deregisterTMCloneTable>
    277a:	48 85 c0             	test   %rax,%rax
    277d:	74 09                	je     2788 <deregister_tm_clones+0x28>
    277f:	ff e0                	jmpq   *%rax
    2781:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2788:	c3                   	retq   
    2789:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002790 <register_tm_clones>:
    2790:	48 8d 3d 59 39 00 00 	lea    0x3959(%rip),%rdi        # 60f0 <__TMC_END__>
    2797:	48 8d 35 52 39 00 00 	lea    0x3952(%rip),%rsi        # 60f0 <__TMC_END__>
    279e:	48 29 fe             	sub    %rdi,%rsi
    27a1:	48 c1 fe 03          	sar    $0x3,%rsi
    27a5:	48 89 f0             	mov    %rsi,%rax
    27a8:	48 c1 e8 3f          	shr    $0x3f,%rax
    27ac:	48 01 c6             	add    %rax,%rsi
    27af:	48 d1 fe             	sar    %rsi
    27b2:	74 14                	je     27c8 <register_tm_clones+0x38>
    27b4:	48 8b 05 35 38 00 00 	mov    0x3835(%rip),%rax        # 5ff0 <_ITM_registerTMCloneTable>
    27bb:	48 85 c0             	test   %rax,%rax
    27be:	74 08                	je     27c8 <register_tm_clones+0x38>
    27c0:	ff e0                	jmpq   *%rax
    27c2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    27c8:	c3                   	retq   
    27c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000027d0 <__do_global_dtors_aux>:
    27d0:	80 3d 61 3b 00 00 00 	cmpb   $0x0,0x3b61(%rip)        # 6338 <completed.7389>
    27d7:	75 2f                	jne    2808 <__do_global_dtors_aux+0x38>
    27d9:	55                   	push   %rbp
    27da:	48 83 3d ee 37 00 00 	cmpq   $0x0,0x37ee(%rip)        # 5fd0 <__cxa_finalize@GLIBC_2.2.5>
    27e1:	00 
    27e2:	48 89 e5             	mov    %rsp,%rbp
    27e5:	74 0c                	je     27f3 <__do_global_dtors_aux+0x23>
    27e7:	48 8b 3d f2 38 00 00 	mov    0x38f2(%rip),%rdi        # 60e0 <__dso_handle>
    27ee:	e8 bd e9 ff ff       	callq  11b0 <__cxa_finalize@plt>
    27f3:	e8 68 ff ff ff       	callq  2760 <deregister_tm_clones>
    27f8:	c6 05 39 3b 00 00 01 	movb   $0x1,0x3b39(%rip)        # 6338 <completed.7389>
    27ff:	5d                   	pop    %rbp
    2800:	c3                   	retq   
    2801:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2808:	c3                   	retq   
    2809:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002810 <frame_dummy>:
    2810:	e9 7b ff ff ff       	jmpq   2790 <register_tm_clones>
    2815:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    281c:	00 00 00 
    281f:	90                   	nop

0000000000002820 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5>:
    2820:	41 57                	push   %r15
    2822:	49 89 f7             	mov    %rsi,%r15
    2825:	41 56                	push   %r14
    2827:	41 55                	push   %r13
    2829:	41 54                	push   %r12
    282b:	55                   	push   %rbp
    282c:	53                   	push   %rbx
    282d:	48 83 ec 18          	sub    $0x18,%rsp
    2831:	4c 63 6e 04          	movslq 0x4(%rsi),%r13
    2835:	49 81 fd fc ff ff 7f 	cmp    $0x7ffffffc,%r13
    283c:	0f 87 8e 00 00 00    	ja     28d0 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5+0xb0>
    2842:	4d 8d 45 01          	lea    0x1(%r13),%r8
    2846:	b8 fd ff ff 7f       	mov    $0x7ffffffd,%eax
    284b:	31 d2                	xor    %edx,%edx
    284d:	48 8b 0d 8c 6b 00 00 	mov    0x6b8c(%rip),%rcx        # 93e0 <_ZL9generator>
    2854:	49 f7 f0             	div    %r8
    2857:	49 ba 05 00 00 00 02 	movabs $0x200000005,%r10
    285e:	00 00 00 
    2861:	4c 0f af c0          	imul   %rax,%r8
    2865:	49 89 c1             	mov    %rax,%r9
    2868:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    286f:	00 
    2870:	48 69 f9 a7 41 00 00 	imul   $0x41a7,%rcx,%rdi
    2877:	48 89 f8             	mov    %rdi,%rax
    287a:	48 89 f9             	mov    %rdi,%rcx
    287d:	49 f7 e2             	mul    %r10
    2880:	48 29 d1             	sub    %rdx,%rcx
    2883:	48 d1 e9             	shr    %rcx
    2886:	48 01 ca             	add    %rcx,%rdx
    2889:	48 c1 ea 1e          	shr    $0x1e,%rdx
    288d:	48 89 d6             	mov    %rdx,%rsi
    2890:	48 c1 e6 1f          	shl    $0x1f,%rsi
    2894:	48 29 d6             	sub    %rdx,%rsi
    2897:	48 29 f7             	sub    %rsi,%rdi
    289a:	48 8d 47 ff          	lea    -0x1(%rdi),%rax
    289e:	48 89 f9             	mov    %rdi,%rcx
    28a1:	49 39 c0             	cmp    %rax,%r8
    28a4:	76 ca                	jbe    2870 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5+0x50>
    28a6:	31 d2                	xor    %edx,%edx
    28a8:	48 89 3d 31 6b 00 00 	mov    %rdi,0x6b31(%rip)        # 93e0 <_ZL9generator>
    28af:	49 f7 f1             	div    %r9
    28b2:	41 03 07             	add    (%r15),%eax
    28b5:	48 83 c4 18          	add    $0x18,%rsp
    28b9:	5b                   	pop    %rbx
    28ba:	5d                   	pop    %rbp
    28bb:	41 5c                	pop    %r12
    28bd:	41 5d                	pop    %r13
    28bf:	41 5e                	pop    %r14
    28c1:	41 5f                	pop    %r15
    28c3:	c3                   	retq   
    28c4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    28cb:	00 00 00 00 
    28cf:	90                   	nop
    28d0:	49 81 fd fd ff ff 7f 	cmp    $0x7ffffffd,%r13
    28d7:	0f 84 a3 00 00 00    	je     2980 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5+0x160>
    28dd:	4c 89 ea             	mov    %r13,%rdx
    28e0:	49 89 fc             	mov    %rdi,%r12
    28e3:	48 bb 09 00 00 00 02 	movabs $0x8000000200000009,%rbx
    28ea:	00 00 80 
    28ed:	48 d1 ea             	shr    %rdx
    28f0:	4c 8d 74 24 08       	lea    0x8(%rsp),%r14
    28f5:	48 8d 2d e4 6a 00 00 	lea    0x6ae4(%rip),%rbp        # 93e0 <_ZL9generator>
    28fc:	48 89 d0             	mov    %rdx,%rax
    28ff:	48 f7 e3             	mul    %rbx
    2902:	48 89 d3             	mov    %rdx,%rbx
    2905:	48 c1 eb 1d          	shr    $0x1d,%rbx
    2909:	4c 89 f6             	mov    %r14,%rsi
    290c:	4c 89 e7             	mov    %r12,%rdi
    290f:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
    2916:	00 
    2917:	89 5c 24 0c          	mov    %ebx,0xc(%rsp)
    291b:	e8 00 ff ff ff       	callq  2820 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5>
    2920:	48 69 7d 00 a7 41 00 	imul   $0x41a7,0x0(%rbp),%rdi
    2927:	00 
    2928:	89 c6                	mov    %eax,%esi
    292a:	48 b8 05 00 00 00 02 	movabs $0x200000005,%rax
    2931:	00 00 00 
    2934:	48 f7 e7             	mul    %rdi
    2937:	48 89 f8             	mov    %rdi,%rax
    293a:	48 29 d0             	sub    %rdx,%rax
    293d:	48 d1 e8             	shr    %rax
    2940:	48 01 c2             	add    %rax,%rdx
    2943:	48 c1 ea 1e          	shr    $0x1e,%rdx
    2947:	48 89 d0             	mov    %rdx,%rax
    294a:	48 c1 e0 1f          	shl    $0x1f,%rax
    294e:	48 29 d0             	sub    %rdx,%rax
    2951:	48 29 c7             	sub    %rax,%rdi
    2954:	48 63 c6             	movslq %esi,%rax
    2957:	48 69 c0 fe ff ff 7f 	imul   $0x7ffffffe,%rax,%rax
    295e:	48 89 fa             	mov    %rdi,%rdx
    2961:	48 89 7d 00          	mov    %rdi,0x0(%rbp)
    2965:	48 ff ca             	dec    %rdx
    2968:	48 01 d0             	add    %rdx,%rax
    296b:	0f 92 c2             	setb   %dl
    296e:	0f b6 d2             	movzbl %dl,%edx
    2971:	49 39 c5             	cmp    %rax,%r13
    2974:	72 93                	jb     2909 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5+0xe9>
    2976:	48 85 d2             	test   %rdx,%rdx
    2979:	75 8e                	jne    2909 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5+0xe9>
    297b:	e9 32 ff ff ff       	jmpq   28b2 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5+0x92>
    2980:	48 69 0d 55 6a 00 00 	imul   $0x41a7,0x6a55(%rip),%rcx        # 93e0 <_ZL9generator>
    2987:	a7 41 00 00 
    298b:	48 ba 05 00 00 00 02 	movabs $0x200000005,%rdx
    2992:	00 00 00 
    2995:	48 89 c8             	mov    %rcx,%rax
    2998:	48 f7 e2             	mul    %rdx
    299b:	48 89 c8             	mov    %rcx,%rax
    299e:	48 29 d0             	sub    %rdx,%rax
    29a1:	48 d1 e8             	shr    %rax
    29a4:	48 01 d0             	add    %rdx,%rax
    29a7:	48 c1 e8 1e          	shr    $0x1e,%rax
    29ab:	48 89 c2             	mov    %rax,%rdx
    29ae:	48 c1 e2 1f          	shl    $0x1f,%rdx
    29b2:	48 29 c2             	sub    %rax,%rdx
    29b5:	48 29 d1             	sub    %rdx,%rcx
    29b8:	48 89 c8             	mov    %rcx,%rax
    29bb:	48 89 0d 1e 6a 00 00 	mov    %rcx,0x6a1e(%rip)        # 93e0 <_ZL9generator>
    29c2:	48 ff c8             	dec    %rax
    29c5:	e9 e8 fe ff ff       	jmpq   28b2 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5+0x92>
    29ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000029d0 <_ZNSt24uniform_int_distributionIhEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEhRT_RKNS0_10param_typeE.constprop.4>:
    29d0:	44 0f b6 4e 01       	movzbl 0x1(%rsi),%r9d
    29d5:	b8 fd ff ff 7f       	mov    $0x7ffffffd,%eax
    29da:	31 d2                	xor    %edx,%edx
    29dc:	48 8b 0d fd 69 00 00 	mov    0x69fd(%rip),%rcx        # 93e0 <_ZL9generator>
    29e3:	49 bb 05 00 00 00 02 	movabs $0x200000005,%r11
    29ea:	00 00 00 
    29ed:	49 ff c1             	inc    %r9
    29f0:	49 f7 f1             	div    %r9
    29f3:	4c 0f af c8          	imul   %rax,%r9
    29f7:	49 89 c2             	mov    %rax,%r10
    29fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2a00:	4c 69 c1 a7 41 00 00 	imul   $0x41a7,%rcx,%r8
    2a07:	4c 89 c0             	mov    %r8,%rax
    2a0a:	4c 89 c1             	mov    %r8,%rcx
    2a0d:	49 f7 e3             	mul    %r11
    2a10:	48 29 d1             	sub    %rdx,%rcx
    2a13:	48 d1 e9             	shr    %rcx
    2a16:	48 01 ca             	add    %rcx,%rdx
    2a19:	48 c1 ea 1e          	shr    $0x1e,%rdx
    2a1d:	48 89 d7             	mov    %rdx,%rdi
    2a20:	48 c1 e7 1f          	shl    $0x1f,%rdi
    2a24:	48 29 d7             	sub    %rdx,%rdi
    2a27:	49 29 f8             	sub    %rdi,%r8
    2a2a:	49 8d 40 ff          	lea    -0x1(%r8),%rax
    2a2e:	4c 89 c1             	mov    %r8,%rcx
    2a31:	49 39 c1             	cmp    %rax,%r9
    2a34:	76 ca                	jbe    2a00 <_ZNSt24uniform_int_distributionIhEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEhRT_RKNS0_10param_typeE.constprop.4+0x30>
    2a36:	31 d2                	xor    %edx,%edx
    2a38:	4c 89 05 a1 69 00 00 	mov    %r8,0x69a1(%rip)        # 93e0 <_ZL9generator>
    2a3f:	49 f7 f2             	div    %r10
    2a42:	02 06                	add    (%rsi),%al
    2a44:	c3                   	retq   
    2a45:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    2a4c:	00 00 00 00 

0000000000002a50 <_Z9descargarPiS_._omp_fn.1>:
    2a50:	55                   	push   %rbp
    2a51:	48 89 e5             	mov    %rsp,%rbp
    2a54:	41 57                	push   %r15
    2a56:	41 56                	push   %r14
    2a58:	41 55                	push   %r13
    2a5a:	41 54                	push   %r12
    2a5c:	53                   	push   %rbx
    2a5d:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    2a61:	4c 63 5f 10          	movslq 0x10(%rdi),%r11
    2a65:	4c 8b 57 08          	mov    0x8(%rdi),%r10
    2a69:	c5 fd 6f 25 4f 69 00 	vmovdqa 0x694f(%rip),%ymm4        # 93c0 <_ZL6zeroes>
    2a70:	00 
    2a71:	c5 fd 6f 1d 27 69 00 	vmovdqa 0x6927(%rip),%ymm3        # 93a0 <_ZL4ones>
    2a78:	00 
    2a79:	44 89 da             	mov    %r11d,%edx
    2a7c:	41 8d 83 f7 3f 00 00 	lea    0x3ff7(%r11),%eax
    2a83:	4c 8b 2f             	mov    (%rdi),%r13
    2a86:	4d 89 d8             	mov    %r11,%r8
    2a89:	4a 8d 1c 9d 00 00 00 	lea    0x0(,%r11,4),%rbx
    2a90:	00 
    2a91:	c5 7d 6f cb          	vmovdqa %ymm3,%ymm9
    2a95:	c5 fd 6f f4          	vmovdqa %ymm4,%ymm6
    2a99:	83 ea 08             	sub    $0x8,%edx
    2a9c:	4d 8d 24 1a          	lea    (%r10,%rbx,1),%r12
    2aa0:	0f 49 c2             	cmovns %edx,%eax
    2aa3:	41 8d bb 00 40 00 00 	lea    0x4000(%r11),%edi
    2aaa:	c5 7d 6f 05 ce 48 00 	vmovdqa 0x48ce(%rip),%ymm8        # 7380 <_ZL8maskfff0>
    2ab1:	00 
    2ab2:	c4 41 7d 6f 14 24    	vmovdqa (%r12),%ymm10
    2ab8:	c5 fd 6f 3d a0 48 00 	vmovdqa 0x48a0(%rip),%ymm7        # 7360 <_ZL8mask000f>
    2abf:	00 
    2ac0:	c1 f8 0e             	sar    $0xe,%eax
    2ac3:	81 ff f8 ff 0f 00    	cmp    $0xffff8,%edi
    2ac9:	c5 ad 66 c3          	vpcmpgtd %ymm3,%ymm10,%ymm0
    2acd:	c5 ad 66 cc          	vpcmpgtd %ymm4,%ymm10,%ymm1
    2ad1:	89 44 24 fc          	mov    %eax,-0x4(%rsp)
    2ad5:	b8 f8 ff 0f 00       	mov    $0xffff8,%eax
    2ada:	0f 4d f8             	cmovge %eax,%edi
    2add:	c5 7d 6f d8          	vmovdqa %ymm0,%ymm11
    2ae1:	c5 fd db c1          	vpand  %ymm1,%ymm0,%ymm0
    2ae5:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    2ae9:	85 c0                	test   %eax,%eax
    2aeb:	0f 84 af 02 00 00    	je     2da0 <_Z9descargarPiS_._omp_fn.1+0x350>
    2af1:	48 8b 0d e8 68 00 00 	mov    0x68e8(%rip),%rcx        # 93e0 <_ZL9generator>
    2af8:	c5 fd 6f d4          	vmovdqa %ymm4,%ymm2
    2afc:	c5 7d 6f e4          	vmovdqa %ymm4,%ymm12
    2b00:	4c 8d 0d d9 68 00 00 	lea    0x68d9(%rip),%r9        # 93e0 <_ZL9generator>
    2b07:	4c 8d 3d 92 48 00 00 	lea    0x4892(%rip),%r15        # 73a0 <_ZL4MASK>
    2b0e:	49 be 05 00 00 00 02 	movabs $0x200000005,%r14
    2b15:	00 00 00 
    2b18:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2b1f:	00 
    2b20:	48 69 f1 a7 41 00 00 	imul   $0x41a7,%rcx,%rsi
    2b27:	48 89 f0             	mov    %rsi,%rax
    2b2a:	48 89 f1             	mov    %rsi,%rcx
    2b2d:	49 f7 e6             	mul    %r14
    2b30:	48 29 d1             	sub    %rdx,%rcx
    2b33:	48 d1 e9             	shr    %rcx
    2b36:	48 01 d1             	add    %rdx,%rcx
    2b39:	48 c1 e9 1e          	shr    $0x1e,%rcx
    2b3d:	48 89 c8             	mov    %rcx,%rax
    2b40:	48 c1 e0 1f          	shl    $0x1f,%rax
    2b44:	48 29 c8             	sub    %rcx,%rax
    2b47:	48 29 c6             	sub    %rax,%rsi
    2b4a:	48 89 f1             	mov    %rsi,%rcx
    2b4d:	48 8d 76 ff          	lea    -0x1(%rsi),%rsi
    2b51:	48 81 fe ff fe ff 7f 	cmp    $0x7ffffeff,%rsi
    2b58:	77 c6                	ja     2b20 <_Z9descargarPiS_._omp_fn.1+0xd0>
    2b5a:	48 b8 01 00 04 00 00 	movabs $0x20000040001,%rax
    2b61:	02 00 00 
    2b64:	49 89 09             	mov    %rcx,(%r9)
    2b67:	48 f7 e6             	mul    %rsi
    2b6a:	48 29 d6             	sub    %rdx,%rsi
    2b6d:	48 d1 ee             	shr    %rsi
    2b70:	48 01 f2             	add    %rsi,%rdx
    2b73:	48 c1 ea 16          	shr    $0x16,%rdx
    2b77:	48 63 d2             	movslq %edx,%rdx
    2b7a:	48 c1 e2 05          	shl    $0x5,%rdx
    2b7e:	c4 c1 7d 6f 2c 17    	vmovdqa (%r15,%rdx,1),%ymm5
    2b84:	c5 d5 ef cb          	vpxor  %ymm3,%ymm5,%ymm1
    2b88:	c5 d5 db e8          	vpand  %ymm0,%ymm5,%ymm5
    2b8c:	c5 f5 db c8          	vpand  %ymm0,%ymm1,%ymm1
    2b90:	c5 fd db c3          	vpand  %ymm3,%ymm0,%ymm0
    2b94:	c5 d5 fe d2          	vpaddd %ymm2,%ymm5,%ymm2
    2b98:	c5 ad fa e8          	vpsubd %ymm0,%ymm10,%ymm5
    2b9c:	c4 c1 75 fe cc       	vpaddd %ymm12,%ymm1,%ymm1
    2ba1:	c5 d5 66 c6          	vpcmpgtd %ymm6,%ymm5,%ymm0
    2ba5:	c5 7d 6f d5          	vmovdqa %ymm5,%ymm10
    2ba9:	c5 7d 6f e1          	vmovdqa %ymm1,%ymm12
    2bad:	c5 a5 db c0          	vpand  %ymm0,%ymm11,%ymm0
    2bb1:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    2bb5:	85 c0                	test   %eax,%eax
    2bb7:	0f 85 63 ff ff ff    	jne    2b20 <_Z9descargarPiS_._omp_fn.1+0xd0>
    2bbd:	c4 e3 fd 00 d2 93    	vpermq $0x93,%ymm2,%ymm2
    2bc3:	c4 c1 7d 7f 2c 24    	vmovdqa %ymm5,(%r12)
    2bc9:	c4 c1 6d db c0       	vpand  %ymm8,%ymm2,%ymm0
    2bce:	c5 ed db d7          	vpand  %ymm7,%ymm2,%ymm2
    2bd2:	c5 f5 fe c8          	vpaddd %ymm0,%ymm1,%ymm1
    2bd6:	48 63 74 24 fc       	movslq -0x4(%rsp),%rsi
    2bdb:	48 8d 15 7e 3f 00 00 	lea    0x3f7e(%rip),%rdx        # 6b60 <left_border>
    2be2:	48 89 f0             	mov    %rsi,%rax
    2be5:	48 c1 e0 05          	shl    $0x5,%rax
    2be9:	c5 fd 7f 0c 02       	vmovdqa %ymm1,(%rdx,%rax,1)
    2bee:	41 8d 40 08          	lea    0x8(%r8),%eax
    2bf2:	39 c7                	cmp    %eax,%edi
    2bf4:	0f 8e c6 01 00 00    	jle    2dc0 <_Z9descargarPiS_._omp_fn.1+0x370>
    2bfa:	44 29 c7             	sub    %r8d,%edi
    2bfd:	c4 63 fd 00 d4 93    	vpermq $0x93,%ymm4,%ymm10
    2c03:	49 8d 4c 1a 20       	lea    0x20(%r10,%rbx,1),%rcx
    2c08:	49 b8 01 00 04 00 00 	movabs $0x20000040001,%r8
    2c0f:	02 00 00 
    2c12:	8d 47 f7             	lea    -0x9(%rdi),%eax
    2c15:	c4 41 2d db d8       	vpand  %ymm8,%ymm10,%ymm11
    2c1a:	c5 2d db d7          	vpand  %ymm7,%ymm10,%ymm10
    2c1e:	31 ff                	xor    %edi,%edi
    2c20:	c1 e8 03             	shr    $0x3,%eax
    2c23:	49 8d 04 c3          	lea    (%r11,%rax,8),%rax
    2c27:	4c 8d 1d 72 47 00 00 	lea    0x4772(%rip),%r11        # 73a0 <_ZL4MASK>
    2c2e:	49 8d 5c 82 40       	lea    0x40(%r10,%rax,4),%rbx
    2c33:	49 ba 05 00 00 00 02 	movabs $0x200000005,%r10
    2c3a:	00 00 00 
    2c3d:	0f 1f 00             	nopl   (%rax)
    2c40:	c5 7d 6f 29          	vmovdqa (%rcx),%ymm13
    2c44:	c4 c1 15 66 c9       	vpcmpgtd %ymm9,%ymm13,%ymm1
    2c49:	c5 95 66 c6          	vpcmpgtd %ymm6,%ymm13,%ymm0
    2c4d:	c5 7d 6f f1          	vmovdqa %ymm1,%ymm14
    2c51:	c5 f5 db c8          	vpand  %ymm0,%ymm1,%ymm1
    2c55:	c5 fd d7 c1          	vpmovmskb %ymm1,%eax
    2c59:	85 c0                	test   %eax,%eax
    2c5b:	0f 84 2f 01 00 00    	je     2d90 <_Z9descargarPiS_._omp_fn.1+0x340>
    2c61:	4c 8b 25 78 67 00 00 	mov    0x6778(%rip),%r12        # 93e0 <_ZL9generator>
    2c68:	4c 8d 0d 71 67 00 00 	lea    0x6771(%rip),%r9        # 93e0 <_ZL9generator>
    2c6f:	c5 7d 6f e4          	vmovdqa %ymm4,%ymm12
    2c73:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    2c7a:	00 00 00 00 
    2c7e:	66 90                	xchg   %ax,%ax
    2c80:	4d 69 f4 a7 41 00 00 	imul   $0x41a7,%r12,%r14
    2c87:	4c 89 f0             	mov    %r14,%rax
    2c8a:	4d 89 f4             	mov    %r14,%r12
    2c8d:	49 f7 e2             	mul    %r10
    2c90:	49 29 d4             	sub    %rdx,%r12
    2c93:	49 d1 ec             	shr    %r12
    2c96:	4c 01 e2             	add    %r12,%rdx
    2c99:	48 c1 ea 1e          	shr    $0x1e,%rdx
    2c9d:	48 89 d0             	mov    %rdx,%rax
    2ca0:	48 c1 e0 1f          	shl    $0x1f,%rax
    2ca4:	48 29 d0             	sub    %rdx,%rax
    2ca7:	49 29 c6             	sub    %rax,%r14
    2caa:	4d 89 f4             	mov    %r14,%r12
    2cad:	4d 8d 76 ff          	lea    -0x1(%r14),%r14
    2cb1:	49 81 fe ff fe ff 7f 	cmp    $0x7ffffeff,%r14
    2cb8:	77 c6                	ja     2c80 <_Z9descargarPiS_._omp_fn.1+0x230>
    2cba:	4c 89 f0             	mov    %r14,%rax
    2cbd:	4d 89 21             	mov    %r12,(%r9)
    2cc0:	49 f7 e0             	mul    %r8
    2cc3:	49 29 d6             	sub    %rdx,%r14
    2cc6:	49 d1 ee             	shr    %r14
    2cc9:	4c 01 f2             	add    %r14,%rdx
    2ccc:	48 c1 ea 16          	shr    $0x16,%rdx
    2cd0:	48 63 d2             	movslq %edx,%rdx
    2cd3:	48 c1 e2 05          	shl    $0x5,%rdx
    2cd7:	c4 c1 7d 6f 2c 13    	vmovdqa (%r11,%rdx,1),%ymm5
    2cdd:	c5 d5 ef c3          	vpxor  %ymm3,%ymm5,%ymm0
    2ce1:	c5 d5 db e9          	vpand  %ymm1,%ymm5,%ymm5
    2ce5:	c5 fd db c1          	vpand  %ymm1,%ymm0,%ymm0
    2ce9:	c5 f5 db cb          	vpand  %ymm3,%ymm1,%ymm1
    2ced:	c4 41 55 fe e4       	vpaddd %ymm12,%ymm5,%ymm12
    2cf2:	c5 95 fa e9          	vpsubd %ymm1,%ymm13,%ymm5
    2cf6:	c5 fd fe c2          	vpaddd %ymm2,%ymm0,%ymm0
    2cfa:	c5 d5 66 ce          	vpcmpgtd %ymm6,%ymm5,%ymm1
    2cfe:	c5 7d 6f ed          	vmovdqa %ymm5,%ymm13
    2d02:	c5 fd 6f d0          	vmovdqa %ymm0,%ymm2
    2d06:	c5 8d db c9          	vpand  %ymm1,%ymm14,%ymm1
    2d0a:	c5 fd d7 c1          	vpmovmskb %ymm1,%eax
    2d0e:	85 c0                	test   %eax,%eax
    2d10:	0f 85 6a ff ff ff    	jne    2c80 <_Z9descargarPiS_._omp_fn.1+0x230>
    2d16:	c4 c3 fd 00 d4 93    	vpermq $0x93,%ymm12,%ymm2
    2d1c:	c5 fd 7f 29          	vmovdqa %ymm5,(%rcx)
    2d20:	c4 c1 6d db c8       	vpand  %ymm8,%ymm2,%ymm1
    2d25:	c5 ed db d7          	vpand  %ymm7,%ymm2,%ymm2
    2d29:	c5 fd fe c1          	vpaddd %ymm1,%ymm0,%ymm0
    2d2d:	c5 fd 6f c8          	vmovdqa %ymm0,%ymm1
    2d31:	c4 e2 7d 17 c9       	vptest %ymm1,%ymm1
    2d36:	74 1c                	je     2d54 <_Z9descargarPiS_._omp_fn.1+0x304>
    2d38:	c5 fd fe 41 fc       	vpaddd -0x4(%rcx),%ymm0,%ymm0
    2d3d:	c5 fe 7f 41 fc       	vmovdqu %ymm0,-0x4(%rcx)
    2d42:	c4 c1 7d 66 c1       	vpcmpgtd %ymm9,%ymm0,%ymm0
    2d47:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    2d4b:	f3 0f b8 c0          	popcnt %eax,%eax
    2d4f:	c1 f8 02             	sar    $0x2,%eax
    2d52:	01 c7                	add    %eax,%edi
    2d54:	48 83 c1 20          	add    $0x20,%rcx
    2d58:	48 39 cb             	cmp    %rcx,%rbx
    2d5b:	0f 85 df fe ff ff    	jne    2c40 <_Z9descargarPiS_._omp_fn.1+0x1f0>
    2d61:	48 89 f0             	mov    %rsi,%rax
    2d64:	48 8d 15 f5 35 00 00 	lea    0x35f5(%rip),%rdx        # 6360 <right_border>
    2d6b:	41 89 7c b5 00       	mov    %edi,0x0(%r13,%rsi,4)
    2d70:	48 c1 e0 05          	shl    $0x5,%rax
    2d74:	c5 fd 7f 14 02       	vmovdqa %ymm2,(%rdx,%rax,1)
    2d79:	c5 f8 77             	vzeroupper 
    2d7c:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
    2d80:	5b                   	pop    %rbx
    2d81:	41 5c                	pop    %r12
    2d83:	41 5d                	pop    %r13
    2d85:	41 5e                	pop    %r14
    2d87:	41 5f                	pop    %r15
    2d89:	5d                   	pop    %rbp
    2d8a:	c3                   	retq   
    2d8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2d90:	c5 a5 fe c2          	vpaddd %ymm2,%ymm11,%ymm0
    2d94:	c5 7d 7f d2          	vmovdqa %ymm10,%ymm2
    2d98:	c5 fd 6f c8          	vmovdqa %ymm0,%ymm1
    2d9c:	eb 93                	jmp    2d31 <_Z9descargarPiS_._omp_fn.1+0x2e1>
    2d9e:	66 90                	xchg   %ax,%ax
    2da0:	c4 e3 fd 00 d4 93    	vpermq $0x93,%ymm4,%ymm2
    2da6:	c4 c1 6d db c8       	vpand  %ymm8,%ymm2,%ymm1
    2dab:	c5 ed db d7          	vpand  %ymm7,%ymm2,%ymm2
    2daf:	c5 f5 fe cc          	vpaddd %ymm4,%ymm1,%ymm1
    2db3:	e9 1e fe ff ff       	jmpq   2bd6 <_Z9descargarPiS_._omp_fn.1+0x186>
    2db8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2dbf:	00 
    2dc0:	31 ff                	xor    %edi,%edi
    2dc2:	eb 9d                	jmp    2d61 <_Z9descargarPiS_._omp_fn.1+0x311>
    2dc4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    2dcb:	00 00 00 00 
    2dcf:	90                   	nop

0000000000002dd0 <_Z9descargarPiS_._omp_fn.0>:
    2dd0:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    2dd5:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    2dd9:	41 ff 72 f8          	pushq  -0x8(%r10)
    2ddd:	55                   	push   %rbp
    2dde:	48 89 e5             	mov    %rsp,%rbp
    2de1:	41 57                	push   %r15
    2de3:	41 56                	push   %r14
    2de5:	41 55                	push   %r13
    2de7:	41 54                	push   %r12
    2de9:	41 52                	push   %r10
    2deb:	53                   	push   %rbx
    2dec:	48 89 fb             	mov    %rdi,%rbx
    2def:	48 83 ec 40          	sub    $0x40,%rsp
    2df3:	4c 8b 6f 08          	mov    0x8(%rdi),%r13
    2df7:	4c 8b 27             	mov    (%rdi),%r12
    2dfa:	e8 31 e2 ff ff       	callq  1030 <GOMP_single_start@plt>
    2dff:	84 c0                	test   %al,%al
    2e01:	74 62                	je     2e65 <_Z9descargarPiS_._omp_fn.0+0x95>
    2e03:	41 be 08 00 00 00    	mov    $0x8,%r14d
    2e09:	4c 8d 7d b0          	lea    -0x50(%rbp),%r15
    2e0d:	0f 1f 00             	nopl   (%rax)
    2e10:	48 83 ec 08          	sub    $0x8,%rsp
    2e14:	48 8b 53 18          	mov    0x18(%rbx),%rdx
    2e18:	44 89 75 c0          	mov    %r14d,-0x40(%rbp)
    2e1c:	4c 89 fe             	mov    %r15,%rsi
    2e1f:	6a 00                	pushq  $0x0
    2e21:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    2e27:	41 b8 08 00 00 00    	mov    $0x8,%r8d
    2e2d:	b9 18 00 00 00       	mov    $0x18,%ecx
    2e32:	6a 00                	pushq  $0x0
    2e34:	48 8d 3d 15 fc ff ff 	lea    -0x3eb(%rip),%rdi        # 2a50 <_Z9descargarPiS_._omp_fn.1>
    2e3b:	41 81 c6 00 40 00 00 	add    $0x4000,%r14d
    2e42:	6a 00                	pushq  $0x0
    2e44:	48 89 55 b0          	mov    %rdx,-0x50(%rbp)
    2e48:	31 d2                	xor    %edx,%edx
    2e4a:	4c 89 65 b8          	mov    %r12,-0x48(%rbp)
    2e4e:	e8 fd e1 ff ff       	callq  1050 <GOMP_task@plt>
    2e53:	48 83 c4 20          	add    $0x20,%rsp
    2e57:	41 81 fe 08 00 10 00 	cmp    $0x100008,%r14d
    2e5e:	75 b0                	jne    2e10 <_Z9descargarPiS_._omp_fn.0+0x40>
    2e60:	e8 8b e2 ff ff       	callq  10f0 <GOMP_taskwait@plt>
    2e65:	e8 26 e3 ff ff       	callq  1190 <GOMP_barrier@plt>
    2e6a:	e8 c1 e1 ff ff       	callq  1030 <GOMP_single_start@plt>
    2e6f:	84 c0                	test   %al,%al
    2e71:	0f 84 a9 00 00 00    	je     2f20 <_Z9descargarPiS_._omp_fn.0+0x150>
    2e77:	48 8d 45 b0          	lea    -0x50(%rbp),%rax
    2e7b:	4d 8d bc 24 e0 ff 3f 	lea    0x3fffe0(%r12),%r15
    2e82:	00 
    2e83:	41 be f8 ff 0f 00    	mov    $0xffff8,%r14d
    2e89:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
    2e8d:	eb 30                	jmp    2ebf <_Z9descargarPiS_._omp_fn.0+0xef>
    2e8f:	90                   	nop
    2e90:	43 8b 84 b5 7c 00 c0 	mov    -0x3fff84(%r13,%r14,4),%eax
    2e97:	ff 
    2e98:	41 03 47 fc          	add    -0x4(%r15),%eax
    2e9c:	83 f8 01             	cmp    $0x1,%eax
    2e9f:	48 8b 53 10          	mov    0x10(%rbx),%rdx
    2ea3:	41 89 47 fc          	mov    %eax,-0x4(%r15)
    2ea7:	0f 9f c0             	setg   %al
    2eaa:	49 ff c6             	inc    %r14
    2ead:	49 83 c7 04          	add    $0x4,%r15
    2eb1:	0f b6 c0             	movzbl %al,%eax
    2eb4:	01 02                	add    %eax,(%rdx)
    2eb6:	49 81 fe 00 00 10 00 	cmp    $0x100000,%r14
    2ebd:	74 61                	je     2f20 <_Z9descargarPiS_._omp_fn.0+0x150>
    2ebf:	41 83 3f 01          	cmpl   $0x1,(%r15)
    2ec3:	44 89 75 ac          	mov    %r14d,-0x54(%rbp)
    2ec7:	7e c7                	jle    2e90 <_Z9descargarPiS_._omp_fn.0+0xc0>
    2ec9:	45 31 db             	xor    %r11d,%r11d
    2ecc:	0f 1f 40 00          	nopl   0x0(%rax)
    2ed0:	48 b8 00 00 00 00 01 	movabs $0x100000000,%rax
    2ed7:	00 00 00 
    2eda:	48 8b 7d a0          	mov    -0x60(%rbp),%rdi
    2ede:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
    2ee2:	48 89 fe             	mov    %rdi,%rsi
    2ee5:	e8 36 f9 ff ff       	callq  2820 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.5>
    2eea:	8b 75 ac             	mov    -0x54(%rbp),%esi
    2eed:	85 c0                	test   %eax,%eax
    2eef:	0f 95 c0             	setne  %al
    2ef2:	41 ff c3             	inc    %r11d
    2ef5:	0f b6 c0             	movzbl %al,%eax
    2ef8:	8d 44 46 ff          	lea    -0x1(%rsi,%rax,2),%eax
    2efc:	83 e0 1f             	and    $0x1f,%eax
    2eff:	41 ff 44 85 00       	incl   0x0(%r13,%rax,4)
    2f04:	45 39 1f             	cmp    %r11d,(%r15)
    2f07:	7f c7                	jg     2ed0 <_Z9descargarPiS_._omp_fn.0+0x100>
    2f09:	41 c7 07 00 00 00 00 	movl   $0x0,(%r15)
    2f10:	e9 7b ff ff ff       	jmpq   2e90 <_Z9descargarPiS_._omp_fn.0+0xc0>
    2f15:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    2f1c:	00 00 00 00 
    2f20:	e8 6b e2 ff ff       	callq  1190 <GOMP_barrier@plt>
    2f25:	e8 06 e1 ff ff       	callq  1030 <GOMP_single_start@plt>
    2f2a:	84 c0                	test   %al,%al
    2f2c:	0f 85 7e 01 00 00    	jne    30b0 <_Z9descargarPiS_._omp_fn.0+0x2e0>
    2f32:	e8 69 e2 ff ff       	callq  11a0 <omp_get_num_threads@plt>
    2f37:	41 89 c5             	mov    %eax,%r13d
    2f3a:	e8 01 e2 ff ff       	callq  1140 <omp_get_thread_num@plt>
    2f3f:	89 c1                	mov    %eax,%ecx
    2f41:	b8 3f 00 00 00       	mov    $0x3f,%eax
    2f46:	99                   	cltd   
    2f47:	41 f7 fd             	idiv   %r13d
    2f4a:	39 d1                	cmp    %edx,%ecx
    2f4c:	0f 8c 96 01 00 00    	jl     30e8 <_Z9descargarPiS_._omp_fn.0+0x318>
    2f52:	0f af c8             	imul   %eax,%ecx
    2f55:	31 ff                	xor    %edi,%edi
    2f57:	01 d1                	add    %edx,%ecx
    2f59:	8d 14 08             	lea    (%rax,%rcx,1),%edx
    2f5c:	39 d1                	cmp    %edx,%ecx
    2f5e:	0f 8d 8f 00 00 00    	jge    2ff3 <_Z9descargarPiS_._omp_fn.0+0x223>
    2f64:	4c 63 c1             	movslq %ecx,%r8
    2f67:	8d 51 01             	lea    0x1(%rcx),%edx
    2f6a:	ff c8                	dec    %eax
    2f6c:	48 8b 4b 18          	mov    0x18(%rbx),%rcx
    2f70:	4c 01 c0             	add    %r8,%rax
    2f73:	c1 e2 0e             	shl    $0xe,%edx
    2f76:	c5 fd 6f 0d 22 64 00 	vmovdqa 0x6422(%rip),%ymm1        # 93a0 <_ZL4ones>
    2f7d:	00 
    2f7e:	4c 8d 0d db 33 00 00 	lea    0x33db(%rip),%r9        # 6360 <right_border>
    2f85:	4a 8d 34 81          	lea    (%rcx,%r8,4),%rsi
    2f89:	48 63 d2             	movslq %edx,%rdx
    2f8c:	4c 89 c1             	mov    %r8,%rcx
    2f8f:	48 c1 e0 10          	shl    $0x10,%rax
    2f93:	49 8d 54 94 1c       	lea    0x1c(%r12,%rdx,4),%rdx
    2f98:	48 c1 e1 05          	shl    $0x5,%rcx
    2f9c:	4d 8d 94 04 1c 00 02 	lea    0x2001c(%r12,%rax,1),%r10
    2fa3:	00 
    2fa4:	4c 8d 05 d5 3b 00 00 	lea    0x3bd5(%rip),%r8        # 6b80 <left_border+0x20>
    2fab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2fb0:	c4 c1 7d 6f 14 09    	vmovdqa (%r9,%rcx,1),%ymm2
    2fb6:	c4 c1 6d fe 04 08    	vpaddd (%r8,%rcx,1),%ymm2,%ymm0
    2fbc:	48 83 c6 04          	add    $0x4,%rsi
    2fc0:	48 83 c1 20          	add    $0x20,%rcx
    2fc4:	c5 fd fe 02          	vpaddd (%rdx),%ymm0,%ymm0
    2fc8:	48 81 c2 00 00 01 00 	add    $0x10000,%rdx
    2fcf:	c5 fe 7f 82 00 00 ff 	vmovdqu %ymm0,-0x10000(%rdx)
    2fd6:	ff 
    2fd7:	c5 fd 66 c1          	vpcmpgtd %ymm1,%ymm0,%ymm0
    2fdb:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    2fdf:	f3 0f b8 c0          	popcnt %eax,%eax
    2fe3:	c1 f8 02             	sar    $0x2,%eax
    2fe6:	03 46 fc             	add    -0x4(%rsi),%eax
    2fe9:	01 c7                	add    %eax,%edi
    2feb:	49 39 d2             	cmp    %rdx,%r10
    2fee:	75 c0                	jne    2fb0 <_Z9descargarPiS_._omp_fn.0+0x1e0>
    2ff0:	c5 f8 77             	vzeroupper 
    2ff3:	48 8b 43 10          	mov    0x10(%rbx),%rax
    2ff7:	f0 01 38             	lock add %edi,(%rax)
    2ffa:	e8 91 e1 ff ff       	callq  1190 <GOMP_barrier@plt>
    2fff:	e8 2c e0 ff ff       	callq  1030 <GOMP_single_start@plt>
    3004:	84 c0                	test   %al,%al
    3006:	0f 84 83 00 00 00    	je     308f <_Z9descargarPiS_._omp_fn.0+0x2bf>
    300c:	48 8b 43 10          	mov    0x10(%rbx),%rax
    3010:	31 d2                	xor    %edx,%edx
    3012:	c5 f9 6f 05 26 3b 00 	vmovdqa 0x3b26(%rip),%xmm0        # 6b40 <right_border+0x7e0>
    3019:	00 
    301a:	41 83 bc 24 e0 ff 3f 	cmpl   $0x1,0x3fffe0(%r12)
    3021:	00 01 
    3023:	8b 08                	mov    (%rax),%ecx
    3025:	0f 9f c2             	setg   %dl
    3028:	29 d1                	sub    %edx,%ecx
    302a:	89 08                	mov    %ecx,(%rax)
    302c:	89 ca                	mov    %ecx,%edx
    302e:	31 c9                	xor    %ecx,%ecx
    3030:	41 83 bc 24 dc ff 3f 	cmpl   $0x1,0x3fffdc(%r12)
    3037:	00 01 
    3039:	0f 9f c1             	setg   %cl
    303c:	29 ca                	sub    %ecx,%edx
    303e:	c4 e3 79 16 c1 01    	vpextrd $0x1,%xmm0,%ecx
    3044:	89 10                	mov    %edx,(%rax)
    3046:	c5 f9 7e c2          	vmovd  %xmm0,%edx
    304a:	41 03 94 24 e0 ff 3f 	add    0x3fffe0(%r12),%edx
    3051:	00 
    3052:	41 01 8c 24 dc ff 3f 	add    %ecx,0x3fffdc(%r12)
    3059:	00 
    305a:	83 fa 01             	cmp    $0x1,%edx
    305d:	41 89 94 24 e0 ff 3f 	mov    %edx,0x3fffe0(%r12)
    3064:	00 
    3065:	0f 9f c2             	setg   %dl
    3068:	8b 08                	mov    (%rax),%ecx
    306a:	0f b6 d2             	movzbl %dl,%edx
    306d:	01 d1                	add    %edx,%ecx
    306f:	31 d2                	xor    %edx,%edx
    3071:	89 08                	mov    %ecx,(%rax)
    3073:	41 83 bc 24 dc ff 3f 	cmpl   $0x1,0x3fffdc(%r12)
    307a:	00 01 
    307c:	0f 9f c2             	setg   %dl
    307f:	01 ca                	add    %ecx,%edx
    3081:	48 8b 4b 18          	mov    0x18(%rbx),%rcx
    3085:	89 10                	mov    %edx,(%rax)
    3087:	03 91 fc 00 00 00    	add    0xfc(%rcx),%edx
    308d:	89 10                	mov    %edx,(%rax)
    308f:	e8 fc e0 ff ff       	callq  1190 <GOMP_barrier@plt>
    3094:	48 8d 65 d0          	lea    -0x30(%rbp),%rsp
    3098:	5b                   	pop    %rbx
    3099:	41 5a                	pop    %r10
    309b:	41 5c                	pop    %r12
    309d:	41 5d                	pop    %r13
    309f:	41 5e                	pop    %r14
    30a1:	41 5f                	pop    %r15
    30a3:	5d                   	pop    %rbp
    30a4:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    30a8:	c3                   	retq   
    30a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    30b0:	c5 fd 6f 1d a8 3a 00 	vmovdqa 0x3aa8(%rip),%ymm3        # 6b60 <left_border>
    30b7:	00 
    30b8:	c4 c1 65 fe 44 24 1c 	vpaddd 0x1c(%r12),%ymm3,%ymm0
    30bf:	48 8b 53 10          	mov    0x10(%rbx),%rdx
    30c3:	c4 c1 7e 7f 44 24 1c 	vmovdqu %ymm0,0x1c(%r12)
    30ca:	c5 fd 66 05 ce 62 00 	vpcmpgtd 0x62ce(%rip),%ymm0,%ymm0        # 93a0 <_ZL4ones>
    30d1:	00 
    30d2:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    30d6:	f3 0f b8 c0          	popcnt %eax,%eax
    30da:	c1 f8 02             	sar    $0x2,%eax
    30dd:	f0 01 02             	lock add %eax,(%rdx)
    30e0:	c5 f8 77             	vzeroupper 
    30e3:	e9 4a fe ff ff       	jmpq   2f32 <_Z9descargarPiS_._omp_fn.0+0x162>
    30e8:	ff c0                	inc    %eax
    30ea:	31 d2                	xor    %edx,%edx
    30ec:	e9 61 fe ff ff       	jmpq   2f52 <_Z9descargarPiS_._omp_fn.0+0x182>
    30f1:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    30f8:	00 00 00 
    30fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000003100 <__libc_csu_init>:
    3100:	41 57                	push   %r15
    3102:	41 56                	push   %r14
    3104:	49 89 d7             	mov    %rdx,%r15
    3107:	41 55                	push   %r13
    3109:	41 54                	push   %r12
    310b:	4c 8d 25 6e 2c 00 00 	lea    0x2c6e(%rip),%r12        # 5d80 <__frame_dummy_init_array_entry>
    3112:	55                   	push   %rbp
    3113:	48 8d 2d 7e 2c 00 00 	lea    0x2c7e(%rip),%rbp        # 5d98 <__init_array_end>
    311a:	53                   	push   %rbx
    311b:	41 89 fd             	mov    %edi,%r13d
    311e:	49 89 f6             	mov    %rsi,%r14
    3121:	4c 29 e5             	sub    %r12,%rbp
    3124:	48 83 ec 08          	sub    $0x8,%rsp
    3128:	48 c1 fd 03          	sar    $0x3,%rbp
    312c:	e8 cf de ff ff       	callq  1000 <_init>
    3131:	48 85 ed             	test   %rbp,%rbp
    3134:	74 20                	je     3156 <__libc_csu_init+0x56>
    3136:	31 db                	xor    %ebx,%ebx
    3138:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    313f:	00 
    3140:	4c 89 fa             	mov    %r15,%rdx
    3143:	4c 89 f6             	mov    %r14,%rsi
    3146:	44 89 ef             	mov    %r13d,%edi
    3149:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
    314d:	48 83 c3 01          	add    $0x1,%rbx
    3151:	48 39 dd             	cmp    %rbx,%rbp
    3154:	75 ea                	jne    3140 <__libc_csu_init+0x40>
    3156:	48 83 c4 08          	add    $0x8,%rsp
    315a:	5b                   	pop    %rbx
    315b:	5d                   	pop    %rbp
    315c:	41 5c                	pop    %r12
    315e:	41 5d                	pop    %r13
    3160:	41 5e                	pop    %r14
    3162:	41 5f                	pop    %r15
    3164:	c3                   	retq   
    3165:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    316c:	00 00 00 00 

0000000000003170 <__libc_csu_fini>:
    3170:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000003174 <_fini>:
    3174:	48 83 ec 08          	sub    $0x8,%rsp
    3178:	48 83 c4 08          	add    $0x8,%rsp
    317c:	c3                   	retq   
