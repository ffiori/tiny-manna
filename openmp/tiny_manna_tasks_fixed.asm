
tiny_manna_tasks:     file format elf64-x86-64
tiny_manna_tasks
architecture: i386:x86-64, flags 0x00000150:
HAS_SYMS, DYNAMIC, D_PAGED
start address 0x0000000000002690

Program Header:
    PHDR off    0x0000000000000040 vaddr 0x0000000000000040 paddr 0x0000000000000040 align 2**3
         filesz 0x0000000000000268 memsz 0x0000000000000268 flags r--
  INTERP off    0x00000000000002a8 vaddr 0x00000000000002a8 paddr 0x00000000000002a8 align 2**0
         filesz 0x000000000000001c memsz 0x000000000000001c flags r--
    LOAD off    0x0000000000000000 vaddr 0x0000000000000000 paddr 0x0000000000000000 align 2**12
         filesz 0x0000000000000f38 memsz 0x0000000000000f38 flags r--
    LOAD off    0x0000000000001000 vaddr 0x0000000000001000 paddr 0x0000000000001000 align 2**12
         filesz 0x0000000000001d6d memsz 0x0000000000001d6d flags r-x
    LOAD off    0x0000000000003000 vaddr 0x0000000000003000 paddr 0x0000000000003000 align 2**12
         filesz 0x0000000000000450 memsz 0x0000000000000450 flags r--
    LOAD off    0x0000000000003d80 vaddr 0x0000000000004d80 paddr 0x0000000000004d80 align 2**12
         filesz 0x0000000000000370 memsz 0x0000000000003660 flags rw-
 DYNAMIC off    0x0000000000003da0 vaddr 0x0000000000004da0 paddr 0x0000000000004da0 align 2**3
         filesz 0x0000000000000230 memsz 0x0000000000000230 flags rw-
    NOTE off    0x00000000000002c4 vaddr 0x00000000000002c4 paddr 0x00000000000002c4 align 2**2
         filesz 0x0000000000000044 memsz 0x0000000000000044 flags r--
EH_FRAME off    0x00000000000031c0 vaddr 0x00000000000031c0 paddr 0x00000000000031c0 align 2**2
         filesz 0x000000000000005c memsz 0x000000000000005c flags r--
   STACK off    0x0000000000000000 vaddr 0x0000000000000000 paddr 0x0000000000000000 align 2**4
         filesz 0x0000000000000000 memsz 0x0000000000000000 flags rw-
   RELRO off    0x0000000000003d80 vaddr 0x0000000000004d80 paddr 0x0000000000004d80 align 2**0
         filesz 0x0000000000000280 memsz 0x0000000000000280 flags r--

Dynamic Section:
  NEEDED               libstdc++.so.6
  NEEDED               libm.so.6
  NEEDED               libgomp.so.1
  NEEDED               libgcc_s.so.1
  NEEDED               libpthread.so.0
  NEEDED               libc.so.6
  INIT                 0x0000000000001000
  FINI                 0x0000000000002d64
  INIT_ARRAY           0x0000000000004d80
  INIT_ARRAYSZ         0x0000000000000018
  FINI_ARRAY           0x0000000000004d98
  FINI_ARRAYSZ         0x0000000000000008
  GNU_HASH             0x0000000000000308
  STRTAB               0x0000000000000668
  SYMTAB               0x0000000000000338
  STRSZ                0x00000000000003fa
  SYMENT               0x0000000000000018
  DEBUG                0x0000000000000000
  PLTGOT               0x0000000000005000
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
 13 .text         00001ba2  00000000000011c0  00000000000011c0  000011c0  2**4
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 14 .fini         00000009  0000000000002d64  0000000000002d64  00002d64  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 15 .rodata       000001c0  0000000000003000  0000000000003000  00003000  2**5
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 16 .eh_frame_hdr 0000005c  00000000000031c0  00000000000031c0  000031c0  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 17 .eh_frame     00000200  0000000000003220  0000000000003220  00003220  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 18 .gcc_except_table 00000030  0000000000003420  0000000000003420  00003420  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 19 .init_array   00000018  0000000000004d80  0000000000004d80  00003d80  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 20 .fini_array   00000008  0000000000004d98  0000000000004d98  00003d98  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 21 .dynamic      00000230  0000000000004da0  0000000000004da0  00003da0  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 22 .got          00000030  0000000000004fd0  0000000000004fd0  00003fd0  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 23 .got.plt      000000d8  0000000000005000  0000000000005000  00004000  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 24 .data         00000018  00000000000050d8  00000000000050d8  000040d8  2**3
                  CONTENTS, ALLOC, LOAD, DATA
 25 .bss          000032e0  0000000000005100  0000000000005100  000040f0  2**5
                  ALLOC
 26 .comment      0000001d  0000000000000000  0000000000000000  000040f0  2**0
                  CONTENTS, READONLY
 27 .debug_aranges 00000030  0000000000000000  0000000000000000  0000410d  2**0
                  CONTENTS, READONLY, DEBUGGING
 28 .debug_info   00000064  0000000000000000  0000000000000000  0000413d  2**0
                  CONTENTS, READONLY, DEBUGGING
 29 .debug_abbrev 0000004d  0000000000000000  0000000000000000  000041a1  2**0
                  CONTENTS, READONLY, DEBUGGING
 30 .debug_line   00000077  0000000000000000  0000000000000000  000041ee  2**0
                  CONTENTS, READONLY, DEBUGGING
 31 .debug_str    0000010b  0000000000000000  0000000000000000  00004265  2**0
                  CONTENTS, READONLY, DEBUGGING
 32 .debug_loc    00000059  0000000000000000  0000000000000000  00004370  2**0
                  CONTENTS, READONLY, DEBUGGING
 33 .debug_ranges 00000020  0000000000000000  0000000000000000  000043c9  2**0
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
0000000000002d64 l    d  .fini	0000000000000000              .fini
0000000000003000 l    d  .rodata	0000000000000000              .rodata
00000000000031c0 l    d  .eh_frame_hdr	0000000000000000              .eh_frame_hdr
0000000000003220 l    d  .eh_frame	0000000000000000              .eh_frame
0000000000003420 l    d  .gcc_except_table	0000000000000000              .gcc_except_table
0000000000004d80 l    d  .init_array	0000000000000000              .init_array
0000000000004d98 l    d  .fini_array	0000000000000000              .fini_array
0000000000004da0 l    d  .dynamic	0000000000000000              .dynamic
0000000000004fd0 l    d  .got	0000000000000000              .got
0000000000005000 l    d  .got.plt	0000000000000000              .got.plt
00000000000050d8 l    d  .data	0000000000000000              .data
0000000000005100 l    d  .bss	0000000000000000              .bss
0000000000000000 l    d  .comment	0000000000000000              .comment
0000000000000000 l    d  .debug_aranges	0000000000000000              .debug_aranges
0000000000000000 l    d  .debug_info	0000000000000000              .debug_info
0000000000000000 l    d  .debug_abbrev	0000000000000000              .debug_abbrev
0000000000000000 l    d  .debug_line	0000000000000000              .debug_line
0000000000000000 l    d  .debug_str	0000000000000000              .debug_str
0000000000000000 l    d  .debug_loc	0000000000000000              .debug_loc
0000000000000000 l    d  .debug_ranges	0000000000000000              .debug_ranges
0000000000000000 l    df *ABS*	0000000000000000              
0000000000002780 l     F .text	0000000000000264              _Z9descargarPiS_._omp_fn.1
00000000000083c0 l     O .bss	0000000000000020              _ZL6zeroes
00000000000083a0 l     O .bss	0000000000000020              _ZL4ones
0000000000006380 l     O .bss	0000000000000020              _ZL8maskfff0
0000000000006360 l     O .bss	0000000000000020              _ZL8mask000f
00000000000063a0 l     O .bss	0000000000002000              _ZL4MASK
0000000000005b60 l     O .bss	0000000000000800              left_border
0000000000005360 l     O .bss	0000000000000800              right_border
00000000000029f0 l     F .text	00000000000002fe              _Z9descargarPiS_._omp_fn.0
00000000000011c0 l     F .text	0000000000000e6f              _GLOBAL__sub_I__Z8randinitv
0000000000005340 l     O .bss	0000000000000001              _ZStL8__ioinit
0000000000000000 l    df *ABS*	0000000000000000              crtfastmath.c
0000000000002670 l     F .text	0000000000000013              set_fast_math
0000000000000000 l    df *ABS*	0000000000000000              crtstuff.c
00000000000026c0 l     F .text	0000000000000000              deregister_tm_clones
00000000000026f0 l     F .text	0000000000000000              register_tm_clones
0000000000002730 l     F .text	0000000000000000              __do_global_dtors_aux
0000000000005338 l     O .bss	0000000000000001              completed.7389
0000000000004d98 l     O .fini_array	0000000000000000              __do_global_dtors_aux_fini_array_entry
0000000000002770 l     F .text	0000000000000000              frame_dummy
0000000000004d80 l     O .init_array	0000000000000000              __frame_dummy_init_array_entry
0000000000000000 l    df *ABS*	0000000000000000              offloadstuff.c
0000000000000000 l    df *ABS*	0000000000000000              crtstuff.c
000000000000341c l     O .eh_frame	0000000000000000              __FRAME_END__
0000000000000000 l    df *ABS*	0000000000000000              offloadstuff.c
0000000000000000 l    df *ABS*	0000000000000000              
00000000000031c0 l       .eh_frame_hdr	0000000000000000              __GNU_EH_FRAME_HDR
0000000000004da0 l     O .dynamic	0000000000000000              _DYNAMIC
0000000000004d98 l       .init_array	0000000000000000              __init_array_end
0000000000004d80 l       .init_array	0000000000000000              __init_array_start
0000000000005000 l     O .got.plt	0000000000000000              _GLOBAL_OFFSET_TABLE_
0000000000000000       F *UND*	0000000000000000              GOMP_single_start@@GOMP_1.0
00000000000050f0 g       .data	0000000000000000              _edata
00000000000031c0 g     O .eh_frame_hdr	0000000000000000              .hidden __offload_funcs_end
00000000000050d8  w      .data	0000000000000000              data_start
0000000000003000 g     O .rodata	0000000000000004              _IO_stdin_used
0000000000000000       F *UND*	0000000000000000              _ZNSt8ios_base15sync_with_stdioEb@@GLIBCXX_3.4
0000000000000000  w    F *UND*	0000000000000000              __cxa_finalize@@GLIBC_2.2.5
0000000000002030 g     F .text	000000000000063e              main
0000000000000000       F *UND*	0000000000000000              GOMP_task@@GOMP_2.0
00000000000050e0 g     O .data	0000000000000000              .hidden __dso_handle
00000000000031c0 g     O .eh_frame_hdr	0000000000000000              .hidden __offload_vars_end
00000000000050e8  w    O .data	0000000000000008              .hidden DW.ref.__gxx_personality_v0
0000000000000000       F *UND*	0000000000000000              _ZNSo5flushEv@@GLIBCXX_3.4
00000000000031c0 g     O .eh_frame_hdr	0000000000000000              .hidden __offload_func_table
0000000000002d64 g     F .fini	0000000000000000              _fini
0000000000000000       F *UND*	0000000000000000              aligned_alloc@@GLIBC_2.16
0000000000000000       F *UND*	0000000000000000              __cxa_atexit@@GLIBC_2.2.5
0000000000000000       F *UND*	0000000000000000              _ZNSt13random_device7_M_finiEv@@GLIBCXX_3.4.18
0000000000000000       F *UND*	0000000000000000              _ZdlPv@@GLIBCXX_3.4
0000000000002690 g     F .text	000000000000002b              _start
0000000000000000       F *UND*	0000000000000000              _ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@@GLIBCXX_3.4
0000000000000000       F *UND*	0000000000000000              _Znwm@@GLIBCXX_3.4
0000000000000000       F *UND*	0000000000000000              _ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@@GLIBCXX_3.4
0000000000001000 g     F .init	0000000000000000              _init
00000000000050f0 g     O .data	0000000000000000              .hidden __TMC_END__
0000000000000000       F *UND*	0000000000000000              _ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@@GLIBCXX_3.4.9
0000000000005100 g     O .bss	0000000000000110              _ZSt4cout@@GLIBCXX_3.4
00000000000050d8 g       .data	0000000000000000              __data_start
0000000000000000       F *UND*	0000000000000000              GOMP_taskwait@@GOMP_2.0
00000000000083e0 g       .bss	0000000000000000              _end
00000000000031c0 g     O .eh_frame_hdr	0000000000000000              .hidden __offload_var_table
0000000000000000       F *UND*	0000000000000000              _ZNSt13random_device9_M_getvalEv@@GLIBCXX_3.4.18
0000000000000000       F *UND*	0000000000000000              _ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@@GLIBCXX_3.4
00000000000050f0 g       .bss	0000000000000000              __bss_start
0000000000000000       F *UND*	0000000000000000              GOMP_parallel@@GOMP_4.0
0000000000000000       F *UND*	0000000000000000              _ZNSt8ios_base4InitC1Ev@@GLIBCXX_3.4
0000000000002cf0 g     F .text	0000000000000065              __libc_csu_init
0000000000000000       F *UND*	0000000000000000              omp_get_thread_num@@OMP_1.0
0000000000000000       F *UND*	0000000000000000              memmove@@GLIBC_2.2.5
0000000000000000       F *UND*	0000000000000000              __gxx_personality_v0@@CXXABI_1.3
0000000000000000       F *UND*	0000000000000000              _ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@@GLIBCXX_3.4.21
0000000000000000       F *UND*	0000000000000000              _ZNSolsEi@@GLIBCXX_3.4
0000000000000000  w      *UND*	0000000000000000              _ITM_deregisterTMCloneTable
0000000000000000       F *UND*	0000000000000000              _Unwind_Resume@@GCC_3.0
0000000000002d60 g     F .text	0000000000000002              __libc_csu_fini
0000000000000000       F *UND*	0000000000000000              GOMP_barrier@@GOMP_1.0
0000000000005220 g     O .bss	0000000000000118              _ZSt3cin@@GLIBCXX_3.4
0000000000000000       F *UND*	0000000000000000              __libc_start_main@@GLIBC_2.2.5
0000000000000000       F *UND*	0000000000000000              omp_get_num_threads@@OMP_1.0
0000000000000000  w      *UND*	0000000000000000              __gmon_start__
0000000000000000  w      *UND*	0000000000000000              _ITM_registerTMCloneTable
0000000000000000       F *UND*	0000000000000000              _ZNSt8ios_base4InitD1Ev@@GLIBCXX_3.4



Disassembly of section .init:

0000000000001000 <_init>:
    1000:	48 83 ec 08          	sub    $0x8,%rsp
    1004:	48 8b 05 dd 3f 00 00 	mov    0x3fdd(%rip),%rax        # 4fe8 <__gmon_start__>
    100b:	48 85 c0             	test   %rax,%rax
    100e:	74 02                	je     1012 <_init+0x12>
    1010:	ff d0                	callq  *%rax
    1012:	48 83 c4 08          	add    $0x8,%rsp
    1016:	c3                   	retq   

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	ff 35 e2 3f 00 00    	pushq  0x3fe2(%rip)        # 5008 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	ff 25 e4 3f 00 00    	jmpq   *0x3fe4(%rip)        # 5010 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000001030 <GOMP_single_start@plt>:
    1030:	ff 25 e2 3f 00 00    	jmpq   *0x3fe2(%rip)        # 5018 <GOMP_single_start@GOMP_1.0>
    1036:	68 00 00 00 00       	pushq  $0x0
    103b:	e9 e0 ff ff ff       	jmpq   1020 <.plt>

0000000000001040 <_ZNSt8ios_base15sync_with_stdioEb@plt>:
    1040:	ff 25 da 3f 00 00    	jmpq   *0x3fda(%rip)        # 5020 <_ZNSt8ios_base15sync_with_stdioEb@GLIBCXX_3.4>
    1046:	68 01 00 00 00       	pushq  $0x1
    104b:	e9 d0 ff ff ff       	jmpq   1020 <.plt>

0000000000001050 <GOMP_task@plt>:
    1050:	ff 25 d2 3f 00 00    	jmpq   *0x3fd2(%rip)        # 5028 <GOMP_task@GOMP_2.0>
    1056:	68 02 00 00 00       	pushq  $0x2
    105b:	e9 c0 ff ff ff       	jmpq   1020 <.plt>

0000000000001060 <_ZNSo5flushEv@plt>:
    1060:	ff 25 ca 3f 00 00    	jmpq   *0x3fca(%rip)        # 5030 <_ZNSo5flushEv@GLIBCXX_3.4>
    1066:	68 03 00 00 00       	pushq  $0x3
    106b:	e9 b0 ff ff ff       	jmpq   1020 <.plt>

0000000000001070 <aligned_alloc@plt>:
    1070:	ff 25 c2 3f 00 00    	jmpq   *0x3fc2(%rip)        # 5038 <aligned_alloc@GLIBC_2.16>
    1076:	68 04 00 00 00       	pushq  $0x4
    107b:	e9 a0 ff ff ff       	jmpq   1020 <.plt>

0000000000001080 <__cxa_atexit@plt>:
    1080:	ff 25 ba 3f 00 00    	jmpq   *0x3fba(%rip)        # 5040 <__cxa_atexit@GLIBC_2.2.5>
    1086:	68 05 00 00 00       	pushq  $0x5
    108b:	e9 90 ff ff ff       	jmpq   1020 <.plt>

0000000000001090 <_ZNSt13random_device7_M_finiEv@plt>:
    1090:	ff 25 b2 3f 00 00    	jmpq   *0x3fb2(%rip)        # 5048 <_ZNSt13random_device7_M_finiEv@GLIBCXX_3.4.18>
    1096:	68 06 00 00 00       	pushq  $0x6
    109b:	e9 80 ff ff ff       	jmpq   1020 <.plt>

00000000000010a0 <_ZdlPv@plt>:
    10a0:	ff 25 aa 3f 00 00    	jmpq   *0x3faa(%rip)        # 5050 <_ZdlPv@GLIBCXX_3.4>
    10a6:	68 07 00 00 00       	pushq  $0x7
    10ab:	e9 70 ff ff ff       	jmpq   1020 <.plt>

00000000000010b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>:
    10b0:	ff 25 a2 3f 00 00    	jmpq   *0x3fa2(%rip)        # 5058 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@GLIBCXX_3.4>
    10b6:	68 08 00 00 00       	pushq  $0x8
    10bb:	e9 60 ff ff ff       	jmpq   1020 <.plt>

00000000000010c0 <_Znwm@plt>:
    10c0:	ff 25 9a 3f 00 00    	jmpq   *0x3f9a(%rip)        # 5060 <_Znwm@GLIBCXX_3.4>
    10c6:	68 09 00 00 00       	pushq  $0x9
    10cb:	e9 50 ff ff ff       	jmpq   1020 <.plt>

00000000000010d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@plt>:
    10d0:	ff 25 92 3f 00 00    	jmpq   *0x3f92(%rip)        # 5068 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@GLIBCXX_3.4>
    10d6:	68 0a 00 00 00       	pushq  $0xa
    10db:	e9 40 ff ff ff       	jmpq   1020 <.plt>

00000000000010e0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>:
    10e0:	ff 25 8a 3f 00 00    	jmpq   *0x3f8a(%rip)        # 5070 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@GLIBCXX_3.4.9>
    10e6:	68 0b 00 00 00       	pushq  $0xb
    10eb:	e9 30 ff ff ff       	jmpq   1020 <.plt>

00000000000010f0 <GOMP_taskwait@plt>:
    10f0:	ff 25 82 3f 00 00    	jmpq   *0x3f82(%rip)        # 5078 <GOMP_taskwait@GOMP_2.0>
    10f6:	68 0c 00 00 00       	pushq  $0xc
    10fb:	e9 20 ff ff ff       	jmpq   1020 <.plt>

0000000000001100 <_ZNSt13random_device9_M_getvalEv@plt>:
    1100:	ff 25 7a 3f 00 00    	jmpq   *0x3f7a(%rip)        # 5080 <_ZNSt13random_device9_M_getvalEv@GLIBCXX_3.4.18>
    1106:	68 0d 00 00 00       	pushq  $0xd
    110b:	e9 10 ff ff ff       	jmpq   1020 <.plt>

0000000000001110 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>:
    1110:	ff 25 72 3f 00 00    	jmpq   *0x3f72(%rip)        # 5088 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@GLIBCXX_3.4>
    1116:	68 0e 00 00 00       	pushq  $0xe
    111b:	e9 00 ff ff ff       	jmpq   1020 <.plt>

0000000000001120 <GOMP_parallel@plt>:
    1120:	ff 25 6a 3f 00 00    	jmpq   *0x3f6a(%rip)        # 5090 <GOMP_parallel@GOMP_4.0>
    1126:	68 0f 00 00 00       	pushq  $0xf
    112b:	e9 f0 fe ff ff       	jmpq   1020 <.plt>

0000000000001130 <_ZNSt8ios_base4InitC1Ev@plt>:
    1130:	ff 25 62 3f 00 00    	jmpq   *0x3f62(%rip)        # 5098 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
    1136:	68 10 00 00 00       	pushq  $0x10
    113b:	e9 e0 fe ff ff       	jmpq   1020 <.plt>

0000000000001140 <omp_get_thread_num@plt>:
    1140:	ff 25 5a 3f 00 00    	jmpq   *0x3f5a(%rip)        # 50a0 <omp_get_thread_num@OMP_1.0>
    1146:	68 11 00 00 00       	pushq  $0x11
    114b:	e9 d0 fe ff ff       	jmpq   1020 <.plt>

0000000000001150 <memmove@plt>:
    1150:	ff 25 52 3f 00 00    	jmpq   *0x3f52(%rip)        # 50a8 <memmove@GLIBC_2.2.5>
    1156:	68 12 00 00 00       	pushq  $0x12
    115b:	e9 c0 fe ff ff       	jmpq   1020 <.plt>

0000000000001160 <_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@plt>:
    1160:	ff 25 4a 3f 00 00    	jmpq   *0x3f4a(%rip)        # 50b0 <_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@GLIBCXX_3.4.21>
    1166:	68 13 00 00 00       	pushq  $0x13
    116b:	e9 b0 fe ff ff       	jmpq   1020 <.plt>

0000000000001170 <_ZNSolsEi@plt>:
    1170:	ff 25 42 3f 00 00    	jmpq   *0x3f42(%rip)        # 50b8 <_ZNSolsEi@GLIBCXX_3.4>
    1176:	68 14 00 00 00       	pushq  $0x14
    117b:	e9 a0 fe ff ff       	jmpq   1020 <.plt>

0000000000001180 <_Unwind_Resume@plt>:
    1180:	ff 25 3a 3f 00 00    	jmpq   *0x3f3a(%rip)        # 50c0 <_Unwind_Resume@GCC_3.0>
    1186:	68 15 00 00 00       	pushq  $0x15
    118b:	e9 90 fe ff ff       	jmpq   1020 <.plt>

0000000000001190 <GOMP_barrier@plt>:
    1190:	ff 25 32 3f 00 00    	jmpq   *0x3f32(%rip)        # 50c8 <GOMP_barrier@GOMP_1.0>
    1196:	68 16 00 00 00       	pushq  $0x16
    119b:	e9 80 fe ff ff       	jmpq   1020 <.plt>

00000000000011a0 <omp_get_num_threads@plt>:
    11a0:	ff 25 2a 3f 00 00    	jmpq   *0x3f2a(%rip)        # 50d0 <omp_get_num_threads@OMP_1.0>
    11a6:	68 17 00 00 00       	pushq  $0x17
    11ab:	e9 70 fe ff ff       	jmpq   1020 <.plt>

Disassembly of section .plt.got:

00000000000011b0 <__cxa_finalize@plt>:
    11b0:	ff 25 1a 3e 00 00    	jmpq   *0x3e1a(%rip)        # 4fd0 <__cxa_finalize@GLIBC_2.2.5>
    11b6:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

00000000000011c0 <_GLOBAL__sub_I__Z8randinitv>:
    11c0:	55                   	push   %rbp
    11c1:	48 8d 3d 78 41 00 00 	lea    0x4178(%rip),%rdi        # 5340 <_ZStL8__ioinit>
    11c8:	48 89 e5             	mov    %rsp,%rbp
    11cb:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    11cf:	e8 5c ff ff ff       	callq  1130 <_ZNSt8ios_base4InitC1Ev@plt>
    11d4:	48 8b 3d 1d 3e 00 00 	mov    0x3e1d(%rip),%rdi        # 4ff8 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
    11db:	48 8d 15 fe 3e 00 00 	lea    0x3efe(%rip),%rdx        # 50e0 <__dso_handle>
    11e2:	48 8d 35 57 41 00 00 	lea    0x4157(%rip),%rsi        # 5340 <_ZStL8__ioinit>
    11e9:	e8 92 fe ff ff       	callq  1080 <__cxa_atexit@plt>
    11ee:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    11f2:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
    11f6:	c5 fd 7f 05 c2 71 00 	vmovdqa %ymm0,0x71c2(%rip)        # 83c0 <_ZL6zeroes>
    11fd:	00 
    11fe:	c5 fd 6f 05 fa 1e 00 	vmovdqa 0x1efa(%rip),%ymm0        # 3100 <_IO_stdin_used+0x100>
    1205:	00 
    1206:	c5 fd 7f 05 92 71 00 	vmovdqa %ymm0,0x7192(%rip)        # 83a0 <_ZL4ones>
    120d:	00 
    120e:	c5 fd 6f 05 0a 1f 00 	vmovdqa 0x1f0a(%rip),%ymm0        # 3120 <_IO_stdin_used+0x120>
    1215:	00 
    1216:	c5 fd 7f 05 62 51 00 	vmovdqa %ymm0,0x5162(%rip)        # 6380 <_ZL8maskfff0>
    121d:	00 
    121e:	c5 fd 6f 05 1a 1f 00 	vmovdqa 0x1f1a(%rip),%ymm0        # 3140 <_IO_stdin_used+0x140>
    1225:	00 
    1226:	c5 fd 7f 05 32 51 00 	vmovdqa %ymm0,0x5132(%rip)        # 6360 <_ZL8mask000f>
    122d:	00 
    122e:	c5 fd 6f 05 ca 1e 00 	vmovdqa 0x1eca(%rip),%ymm0        # 3100 <_IO_stdin_used+0x100>
    1235:	00 
    1236:	c4 e3 7d 02 d1 01    	vpblendd $0x1,%ymm1,%ymm0,%ymm2
    123c:	c5 fd 7f 05 5c 51 00 	vmovdqa %ymm0,0x515c(%rip)        # 63a0 <_ZL4MASK>
    1243:	00 
    1244:	c5 fd 7f 15 74 51 00 	vmovdqa %ymm2,0x5174(%rip)        # 63c0 <_ZL4MASK+0x20>
    124b:	00 
    124c:	c4 e3 7d 02 d1 02    	vpblendd $0x2,%ymm1,%ymm0,%ymm2
    1252:	c5 fd 7f 15 86 51 00 	vmovdqa %ymm2,0x5186(%rip)        # 63e0 <_ZL4MASK+0x40>
    1259:	00 
    125a:	c4 e3 7d 02 d1 03    	vpblendd $0x3,%ymm1,%ymm0,%ymm2
    1260:	c5 fd 7f 15 98 51 00 	vmovdqa %ymm2,0x5198(%rip)        # 6400 <_ZL4MASK+0x60>
    1267:	00 
    1268:	c4 e3 7d 02 d1 04    	vpblendd $0x4,%ymm1,%ymm0,%ymm2
    126e:	c5 fd 7f 15 aa 51 00 	vmovdqa %ymm2,0x51aa(%rip)        # 6420 <_ZL4MASK+0x80>
    1275:	00 
    1276:	c4 e3 7d 02 d1 05    	vpblendd $0x5,%ymm1,%ymm0,%ymm2
    127c:	c5 fd 7f 15 bc 51 00 	vmovdqa %ymm2,0x51bc(%rip)        # 6440 <_ZL4MASK+0xa0>
    1283:	00 
    1284:	c4 e3 7d 02 d1 06    	vpblendd $0x6,%ymm1,%ymm0,%ymm2
    128a:	c5 fd 7f 15 ce 51 00 	vmovdqa %ymm2,0x51ce(%rip)        # 6460 <_ZL4MASK+0xc0>
    1291:	00 
    1292:	c4 e3 7d 02 d1 07    	vpblendd $0x7,%ymm1,%ymm0,%ymm2
    1298:	c5 fd 7f 15 e0 51 00 	vmovdqa %ymm2,0x51e0(%rip)        # 6480 <_ZL4MASK+0xe0>
    129f:	00 
    12a0:	c4 e3 7d 02 d1 08    	vpblendd $0x8,%ymm1,%ymm0,%ymm2
    12a6:	c5 fd 7f 15 f2 51 00 	vmovdqa %ymm2,0x51f2(%rip)        # 64a0 <_ZL4MASK+0x100>
    12ad:	00 
    12ae:	c4 e3 7d 02 d1 09    	vpblendd $0x9,%ymm1,%ymm0,%ymm2
    12b4:	c5 fd 7f 15 04 52 00 	vmovdqa %ymm2,0x5204(%rip)        # 64c0 <_ZL4MASK+0x120>
    12bb:	00 
    12bc:	c4 e3 7d 02 d1 0a    	vpblendd $0xa,%ymm1,%ymm0,%ymm2
    12c2:	c5 fd 7f 15 16 52 00 	vmovdqa %ymm2,0x5216(%rip)        # 64e0 <_ZL4MASK+0x140>
    12c9:	00 
    12ca:	c4 e3 7d 02 d1 0b    	vpblendd $0xb,%ymm1,%ymm0,%ymm2
    12d0:	c5 fd 7f 15 28 52 00 	vmovdqa %ymm2,0x5228(%rip)        # 6500 <_ZL4MASK+0x160>
    12d7:	00 
    12d8:	c4 e3 7d 02 d1 0c    	vpblendd $0xc,%ymm1,%ymm0,%ymm2
    12de:	c5 fd 7f 15 3a 52 00 	vmovdqa %ymm2,0x523a(%rip)        # 6520 <_ZL4MASK+0x180>
    12e5:	00 
    12e6:	c4 e3 7d 02 d1 0d    	vpblendd $0xd,%ymm1,%ymm0,%ymm2
    12ec:	c5 fd 7f 15 4c 52 00 	vmovdqa %ymm2,0x524c(%rip)        # 6540 <_ZL4MASK+0x1a0>
    12f3:	00 
    12f4:	c4 e3 7d 02 d1 0e    	vpblendd $0xe,%ymm1,%ymm0,%ymm2
    12fa:	c5 fd 7f 15 5e 52 00 	vmovdqa %ymm2,0x525e(%rip)        # 6560 <_ZL4MASK+0x1c0>
    1301:	00 
    1302:	c4 e3 7d 02 d1 0f    	vpblendd $0xf,%ymm1,%ymm0,%ymm2
    1308:	c5 fd 7f 15 70 52 00 	vmovdqa %ymm2,0x5270(%rip)        # 6580 <_ZL4MASK+0x1e0>
    130f:	00 
    1310:	c4 e3 7d 02 d1 10    	vpblendd $0x10,%ymm1,%ymm0,%ymm2
    1316:	c5 fd 7f 15 82 52 00 	vmovdqa %ymm2,0x5282(%rip)        # 65a0 <_ZL4MASK+0x200>
    131d:	00 
    131e:	c4 e3 7d 02 d1 11    	vpblendd $0x11,%ymm1,%ymm0,%ymm2
    1324:	c5 fd 7f 15 94 52 00 	vmovdqa %ymm2,0x5294(%rip)        # 65c0 <_ZL4MASK+0x220>
    132b:	00 
    132c:	c4 e3 7d 02 d1 12    	vpblendd $0x12,%ymm1,%ymm0,%ymm2
    1332:	c5 fd 7f 15 a6 52 00 	vmovdqa %ymm2,0x52a6(%rip)        # 65e0 <_ZL4MASK+0x240>
    1339:	00 
    133a:	c4 e3 7d 02 d1 13    	vpblendd $0x13,%ymm1,%ymm0,%ymm2
    1340:	c5 fd 7f 15 b8 52 00 	vmovdqa %ymm2,0x52b8(%rip)        # 6600 <_ZL4MASK+0x260>
    1347:	00 
    1348:	c4 e3 7d 02 d1 14    	vpblendd $0x14,%ymm1,%ymm0,%ymm2
    134e:	c5 fd 7f 15 ca 52 00 	vmovdqa %ymm2,0x52ca(%rip)        # 6620 <_ZL4MASK+0x280>
    1355:	00 
    1356:	c4 e3 7d 02 d1 15    	vpblendd $0x15,%ymm1,%ymm0,%ymm2
    135c:	c5 fd 7f 15 dc 52 00 	vmovdqa %ymm2,0x52dc(%rip)        # 6640 <_ZL4MASK+0x2a0>
    1363:	00 
    1364:	c4 e3 7d 02 d1 16    	vpblendd $0x16,%ymm1,%ymm0,%ymm2
    136a:	c5 fd 7f 15 ee 52 00 	vmovdqa %ymm2,0x52ee(%rip)        # 6660 <_ZL4MASK+0x2c0>
    1371:	00 
    1372:	c4 e3 7d 02 d1 17    	vpblendd $0x17,%ymm1,%ymm0,%ymm2
    1378:	c5 fd 7f 15 00 53 00 	vmovdqa %ymm2,0x5300(%rip)        # 6680 <_ZL4MASK+0x2e0>
    137f:	00 
    1380:	c4 e3 7d 02 d1 18    	vpblendd $0x18,%ymm1,%ymm0,%ymm2
    1386:	c5 fd 7f 15 12 53 00 	vmovdqa %ymm2,0x5312(%rip)        # 66a0 <_ZL4MASK+0x300>
    138d:	00 
    138e:	c4 e3 7d 02 d1 19    	vpblendd $0x19,%ymm1,%ymm0,%ymm2
    1394:	c5 fd 7f 15 24 53 00 	vmovdqa %ymm2,0x5324(%rip)        # 66c0 <_ZL4MASK+0x320>
    139b:	00 
    139c:	c4 e3 7d 02 d1 1a    	vpblendd $0x1a,%ymm1,%ymm0,%ymm2
    13a2:	c5 fd 7f 15 36 53 00 	vmovdqa %ymm2,0x5336(%rip)        # 66e0 <_ZL4MASK+0x340>
    13a9:	00 
    13aa:	c4 e3 7d 02 d1 1b    	vpblendd $0x1b,%ymm1,%ymm0,%ymm2
    13b0:	c5 fd 7f 15 48 53 00 	vmovdqa %ymm2,0x5348(%rip)        # 6700 <_ZL4MASK+0x360>
    13b7:	00 
    13b8:	c4 e3 7d 02 d1 1c    	vpblendd $0x1c,%ymm1,%ymm0,%ymm2
    13be:	c5 fd 7f 15 5a 53 00 	vmovdqa %ymm2,0x535a(%rip)        # 6720 <_ZL4MASK+0x380>
    13c5:	00 
    13c6:	c4 e3 7d 02 d1 1d    	vpblendd $0x1d,%ymm1,%ymm0,%ymm2
    13cc:	c5 fd 7f 15 6c 53 00 	vmovdqa %ymm2,0x536c(%rip)        # 6740 <_ZL4MASK+0x3a0>
    13d3:	00 
    13d4:	c4 e3 7d 02 d1 1e    	vpblendd $0x1e,%ymm1,%ymm0,%ymm2
    13da:	c5 fd 7f 15 7e 53 00 	vmovdqa %ymm2,0x537e(%rip)        # 6760 <_ZL4MASK+0x3c0>
    13e1:	00 
    13e2:	c4 e3 7d 02 d1 1f    	vpblendd $0x1f,%ymm1,%ymm0,%ymm2
    13e8:	c5 fd 7f 15 90 53 00 	vmovdqa %ymm2,0x5390(%rip)        # 6780 <_ZL4MASK+0x3e0>
    13ef:	00 
    13f0:	c4 e3 7d 02 d1 20    	vpblendd $0x20,%ymm1,%ymm0,%ymm2
    13f6:	c5 fd 7f 15 a2 53 00 	vmovdqa %ymm2,0x53a2(%rip)        # 67a0 <_ZL4MASK+0x400>
    13fd:	00 
    13fe:	c4 e3 7d 02 d1 21    	vpblendd $0x21,%ymm1,%ymm0,%ymm2
    1404:	c5 fd 7f 15 b4 53 00 	vmovdqa %ymm2,0x53b4(%rip)        # 67c0 <_ZL4MASK+0x420>
    140b:	00 
    140c:	c4 e3 7d 02 d1 22    	vpblendd $0x22,%ymm1,%ymm0,%ymm2
    1412:	c5 fd 7f 15 c6 53 00 	vmovdqa %ymm2,0x53c6(%rip)        # 67e0 <_ZL4MASK+0x440>
    1419:	00 
    141a:	c4 e3 7d 02 d1 23    	vpblendd $0x23,%ymm1,%ymm0,%ymm2
    1420:	c5 fd 7f 15 d8 53 00 	vmovdqa %ymm2,0x53d8(%rip)        # 6800 <_ZL4MASK+0x460>
    1427:	00 
    1428:	c4 e3 7d 02 d1 24    	vpblendd $0x24,%ymm1,%ymm0,%ymm2
    142e:	c5 fd 7f 15 ea 53 00 	vmovdqa %ymm2,0x53ea(%rip)        # 6820 <_ZL4MASK+0x480>
    1435:	00 
    1436:	c4 e3 7d 02 d1 25    	vpblendd $0x25,%ymm1,%ymm0,%ymm2
    143c:	c5 fd 7f 15 fc 53 00 	vmovdqa %ymm2,0x53fc(%rip)        # 6840 <_ZL4MASK+0x4a0>
    1443:	00 
    1444:	c4 e3 7d 02 d1 26    	vpblendd $0x26,%ymm1,%ymm0,%ymm2
    144a:	c5 fd 7f 15 0e 54 00 	vmovdqa %ymm2,0x540e(%rip)        # 6860 <_ZL4MASK+0x4c0>
    1451:	00 
    1452:	c4 e3 7d 02 d1 27    	vpblendd $0x27,%ymm1,%ymm0,%ymm2
    1458:	c5 fd 7f 15 20 54 00 	vmovdqa %ymm2,0x5420(%rip)        # 6880 <_ZL4MASK+0x4e0>
    145f:	00 
    1460:	c4 e3 7d 02 d1 28    	vpblendd $0x28,%ymm1,%ymm0,%ymm2
    1466:	c5 fd 7f 15 32 54 00 	vmovdqa %ymm2,0x5432(%rip)        # 68a0 <_ZL4MASK+0x500>
    146d:	00 
    146e:	c4 e3 7d 02 d1 29    	vpblendd $0x29,%ymm1,%ymm0,%ymm2
    1474:	c5 fd 7f 15 44 54 00 	vmovdqa %ymm2,0x5444(%rip)        # 68c0 <_ZL4MASK+0x520>
    147b:	00 
    147c:	c4 e3 7d 02 d1 2a    	vpblendd $0x2a,%ymm1,%ymm0,%ymm2
    1482:	c5 fd 7f 15 56 54 00 	vmovdqa %ymm2,0x5456(%rip)        # 68e0 <_ZL4MASK+0x540>
    1489:	00 
    148a:	c4 e3 7d 02 d1 2b    	vpblendd $0x2b,%ymm1,%ymm0,%ymm2
    1490:	c5 fd 7f 15 68 54 00 	vmovdqa %ymm2,0x5468(%rip)        # 6900 <_ZL4MASK+0x560>
    1497:	00 
    1498:	c4 e3 7d 02 d1 2c    	vpblendd $0x2c,%ymm1,%ymm0,%ymm2
    149e:	c5 fd 7f 15 7a 54 00 	vmovdqa %ymm2,0x547a(%rip)        # 6920 <_ZL4MASK+0x580>
    14a5:	00 
    14a6:	c4 e3 7d 02 d1 2d    	vpblendd $0x2d,%ymm1,%ymm0,%ymm2
    14ac:	c5 fd 7f 15 8c 54 00 	vmovdqa %ymm2,0x548c(%rip)        # 6940 <_ZL4MASK+0x5a0>
    14b3:	00 
    14b4:	c4 e3 7d 02 d1 2e    	vpblendd $0x2e,%ymm1,%ymm0,%ymm2
    14ba:	c5 fd 7f 15 9e 54 00 	vmovdqa %ymm2,0x549e(%rip)        # 6960 <_ZL4MASK+0x5c0>
    14c1:	00 
    14c2:	c4 e3 7d 02 d1 2f    	vpblendd $0x2f,%ymm1,%ymm0,%ymm2
    14c8:	c5 fd 7f 15 b0 54 00 	vmovdqa %ymm2,0x54b0(%rip)        # 6980 <_ZL4MASK+0x5e0>
    14cf:	00 
    14d0:	c4 e3 7d 02 d1 30    	vpblendd $0x30,%ymm1,%ymm0,%ymm2
    14d6:	c5 fd 7f 15 c2 54 00 	vmovdqa %ymm2,0x54c2(%rip)        # 69a0 <_ZL4MASK+0x600>
    14dd:	00 
    14de:	c4 e3 7d 02 d1 31    	vpblendd $0x31,%ymm1,%ymm0,%ymm2
    14e4:	c5 fd 7f 15 d4 54 00 	vmovdqa %ymm2,0x54d4(%rip)        # 69c0 <_ZL4MASK+0x620>
    14eb:	00 
    14ec:	c4 e3 7d 02 d1 32    	vpblendd $0x32,%ymm1,%ymm0,%ymm2
    14f2:	c5 fd 7f 15 e6 54 00 	vmovdqa %ymm2,0x54e6(%rip)        # 69e0 <_ZL4MASK+0x640>
    14f9:	00 
    14fa:	c4 e3 7d 02 d1 33    	vpblendd $0x33,%ymm1,%ymm0,%ymm2
    1500:	c5 fd 7f 15 f8 54 00 	vmovdqa %ymm2,0x54f8(%rip)        # 6a00 <_ZL4MASK+0x660>
    1507:	00 
    1508:	c4 e3 7d 02 d1 34    	vpblendd $0x34,%ymm1,%ymm0,%ymm2
    150e:	c5 fd 7f 15 0a 55 00 	vmovdqa %ymm2,0x550a(%rip)        # 6a20 <_ZL4MASK+0x680>
    1515:	00 
    1516:	c4 e3 7d 02 d1 35    	vpblendd $0x35,%ymm1,%ymm0,%ymm2
    151c:	c5 fd 7f 15 1c 55 00 	vmovdqa %ymm2,0x551c(%rip)        # 6a40 <_ZL4MASK+0x6a0>
    1523:	00 
    1524:	c4 e3 7d 02 d1 36    	vpblendd $0x36,%ymm1,%ymm0,%ymm2
    152a:	c5 fd 7f 15 2e 55 00 	vmovdqa %ymm2,0x552e(%rip)        # 6a60 <_ZL4MASK+0x6c0>
    1531:	00 
    1532:	c4 e3 7d 02 d1 37    	vpblendd $0x37,%ymm1,%ymm0,%ymm2
    1538:	c5 fd 7f 15 40 55 00 	vmovdqa %ymm2,0x5540(%rip)        # 6a80 <_ZL4MASK+0x6e0>
    153f:	00 
    1540:	c4 e3 7d 02 d1 38    	vpblendd $0x38,%ymm1,%ymm0,%ymm2
    1546:	c5 fd 7f 15 52 55 00 	vmovdqa %ymm2,0x5552(%rip)        # 6aa0 <_ZL4MASK+0x700>
    154d:	00 
    154e:	c4 e3 7d 02 d1 39    	vpblendd $0x39,%ymm1,%ymm0,%ymm2
    1554:	c5 fd 7f 15 64 55 00 	vmovdqa %ymm2,0x5564(%rip)        # 6ac0 <_ZL4MASK+0x720>
    155b:	00 
    155c:	c4 e3 7d 02 d1 3a    	vpblendd $0x3a,%ymm1,%ymm0,%ymm2
    1562:	c5 fd 7f 15 76 55 00 	vmovdqa %ymm2,0x5576(%rip)        # 6ae0 <_ZL4MASK+0x740>
    1569:	00 
    156a:	c4 e3 7d 02 d1 3b    	vpblendd $0x3b,%ymm1,%ymm0,%ymm2
    1570:	c5 fd 7f 15 88 55 00 	vmovdqa %ymm2,0x5588(%rip)        # 6b00 <_ZL4MASK+0x760>
    1577:	00 
    1578:	c4 e3 7d 02 d1 3c    	vpblendd $0x3c,%ymm1,%ymm0,%ymm2
    157e:	c5 fd 7f 15 9a 55 00 	vmovdqa %ymm2,0x559a(%rip)        # 6b20 <_ZL4MASK+0x780>
    1585:	00 
    1586:	c4 e3 7d 02 d1 3d    	vpblendd $0x3d,%ymm1,%ymm0,%ymm2
    158c:	c5 fd 7f 15 ac 55 00 	vmovdqa %ymm2,0x55ac(%rip)        # 6b40 <_ZL4MASK+0x7a0>
    1593:	00 
    1594:	c4 e3 7d 02 d1 3e    	vpblendd $0x3e,%ymm1,%ymm0,%ymm2
    159a:	c5 fd 7f 15 be 55 00 	vmovdqa %ymm2,0x55be(%rip)        # 6b60 <_ZL4MASK+0x7c0>
    15a1:	00 
    15a2:	c4 e3 7d 02 d1 3f    	vpblendd $0x3f,%ymm1,%ymm0,%ymm2
    15a8:	c5 fd 7f 15 d0 55 00 	vmovdqa %ymm2,0x55d0(%rip)        # 6b80 <_ZL4MASK+0x7e0>
    15af:	00 
    15b0:	c4 e3 7d 02 d1 40    	vpblendd $0x40,%ymm1,%ymm0,%ymm2
    15b6:	c5 fd 7f 15 e2 55 00 	vmovdqa %ymm2,0x55e2(%rip)        # 6ba0 <_ZL4MASK+0x800>
    15bd:	00 
    15be:	c4 e3 7d 02 d1 41    	vpblendd $0x41,%ymm1,%ymm0,%ymm2
    15c4:	c5 fd 7f 15 f4 55 00 	vmovdqa %ymm2,0x55f4(%rip)        # 6bc0 <_ZL4MASK+0x820>
    15cb:	00 
    15cc:	c4 e3 7d 02 d1 42    	vpblendd $0x42,%ymm1,%ymm0,%ymm2
    15d2:	c5 fd 7f 15 06 56 00 	vmovdqa %ymm2,0x5606(%rip)        # 6be0 <_ZL4MASK+0x840>
    15d9:	00 
    15da:	c4 e3 7d 02 d1 43    	vpblendd $0x43,%ymm1,%ymm0,%ymm2
    15e0:	c5 fd 7f 15 18 56 00 	vmovdqa %ymm2,0x5618(%rip)        # 6c00 <_ZL4MASK+0x860>
    15e7:	00 
    15e8:	c4 e3 7d 02 d1 44    	vpblendd $0x44,%ymm1,%ymm0,%ymm2
    15ee:	c5 fd 7f 15 2a 56 00 	vmovdqa %ymm2,0x562a(%rip)        # 6c20 <_ZL4MASK+0x880>
    15f5:	00 
    15f6:	c4 e3 7d 02 d1 45    	vpblendd $0x45,%ymm1,%ymm0,%ymm2
    15fc:	c5 fd 7f 15 3c 56 00 	vmovdqa %ymm2,0x563c(%rip)        # 6c40 <_ZL4MASK+0x8a0>
    1603:	00 
    1604:	c4 e3 7d 02 d1 46    	vpblendd $0x46,%ymm1,%ymm0,%ymm2
    160a:	c5 fd 7f 15 4e 56 00 	vmovdqa %ymm2,0x564e(%rip)        # 6c60 <_ZL4MASK+0x8c0>
    1611:	00 
    1612:	c4 e3 7d 02 d1 47    	vpblendd $0x47,%ymm1,%ymm0,%ymm2
    1618:	c5 fd 7f 15 60 56 00 	vmovdqa %ymm2,0x5660(%rip)        # 6c80 <_ZL4MASK+0x8e0>
    161f:	00 
    1620:	c4 e3 7d 02 d1 48    	vpblendd $0x48,%ymm1,%ymm0,%ymm2
    1626:	c5 fd 7f 15 72 56 00 	vmovdqa %ymm2,0x5672(%rip)        # 6ca0 <_ZL4MASK+0x900>
    162d:	00 
    162e:	c4 e3 7d 02 d1 49    	vpblendd $0x49,%ymm1,%ymm0,%ymm2
    1634:	c5 fd 7f 15 84 56 00 	vmovdqa %ymm2,0x5684(%rip)        # 6cc0 <_ZL4MASK+0x920>
    163b:	00 
    163c:	c4 e3 7d 02 d1 4a    	vpblendd $0x4a,%ymm1,%ymm0,%ymm2
    1642:	c5 fd 7f 15 96 56 00 	vmovdqa %ymm2,0x5696(%rip)        # 6ce0 <_ZL4MASK+0x940>
    1649:	00 
    164a:	c4 e3 7d 02 d1 4b    	vpblendd $0x4b,%ymm1,%ymm0,%ymm2
    1650:	c5 fd 7f 15 a8 56 00 	vmovdqa %ymm2,0x56a8(%rip)        # 6d00 <_ZL4MASK+0x960>
    1657:	00 
    1658:	c4 e3 7d 02 d1 4c    	vpblendd $0x4c,%ymm1,%ymm0,%ymm2
    165e:	c5 fd 7f 15 ba 56 00 	vmovdqa %ymm2,0x56ba(%rip)        # 6d20 <_ZL4MASK+0x980>
    1665:	00 
    1666:	c4 e3 7d 02 d1 4d    	vpblendd $0x4d,%ymm1,%ymm0,%ymm2
    166c:	c5 fd 7f 15 cc 56 00 	vmovdqa %ymm2,0x56cc(%rip)        # 6d40 <_ZL4MASK+0x9a0>
    1673:	00 
    1674:	c4 e3 7d 02 d1 4e    	vpblendd $0x4e,%ymm1,%ymm0,%ymm2
    167a:	c5 fd 7f 15 de 56 00 	vmovdqa %ymm2,0x56de(%rip)        # 6d60 <_ZL4MASK+0x9c0>
    1681:	00 
    1682:	c4 e3 7d 02 d1 4f    	vpblendd $0x4f,%ymm1,%ymm0,%ymm2
    1688:	c5 fd 7f 15 f0 56 00 	vmovdqa %ymm2,0x56f0(%rip)        # 6d80 <_ZL4MASK+0x9e0>
    168f:	00 
    1690:	c4 e3 7d 02 d1 50    	vpblendd $0x50,%ymm1,%ymm0,%ymm2
    1696:	c5 fd 7f 15 02 57 00 	vmovdqa %ymm2,0x5702(%rip)        # 6da0 <_ZL4MASK+0xa00>
    169d:	00 
    169e:	c4 e3 7d 02 d1 51    	vpblendd $0x51,%ymm1,%ymm0,%ymm2
    16a4:	c5 fd 7f 15 14 57 00 	vmovdqa %ymm2,0x5714(%rip)        # 6dc0 <_ZL4MASK+0xa20>
    16ab:	00 
    16ac:	c4 e3 7d 02 d1 52    	vpblendd $0x52,%ymm1,%ymm0,%ymm2
    16b2:	c5 fd 7f 15 26 57 00 	vmovdqa %ymm2,0x5726(%rip)        # 6de0 <_ZL4MASK+0xa40>
    16b9:	00 
    16ba:	c4 e3 7d 02 d1 53    	vpblendd $0x53,%ymm1,%ymm0,%ymm2
    16c0:	c5 fd 7f 15 38 57 00 	vmovdqa %ymm2,0x5738(%rip)        # 6e00 <_ZL4MASK+0xa60>
    16c7:	00 
    16c8:	c4 e3 7d 02 d1 54    	vpblendd $0x54,%ymm1,%ymm0,%ymm2
    16ce:	c5 fd 7f 15 4a 57 00 	vmovdqa %ymm2,0x574a(%rip)        # 6e20 <_ZL4MASK+0xa80>
    16d5:	00 
    16d6:	c4 e3 7d 02 d1 55    	vpblendd $0x55,%ymm1,%ymm0,%ymm2
    16dc:	c5 fd 7f 15 5c 57 00 	vmovdqa %ymm2,0x575c(%rip)        # 6e40 <_ZL4MASK+0xaa0>
    16e3:	00 
    16e4:	c4 e3 7d 02 d1 56    	vpblendd $0x56,%ymm1,%ymm0,%ymm2
    16ea:	c5 fd 7f 15 6e 57 00 	vmovdqa %ymm2,0x576e(%rip)        # 6e60 <_ZL4MASK+0xac0>
    16f1:	00 
    16f2:	c4 e3 7d 02 d1 57    	vpblendd $0x57,%ymm1,%ymm0,%ymm2
    16f8:	c5 fd 7f 15 80 57 00 	vmovdqa %ymm2,0x5780(%rip)        # 6e80 <_ZL4MASK+0xae0>
    16ff:	00 
    1700:	c4 e3 7d 02 d1 58    	vpblendd $0x58,%ymm1,%ymm0,%ymm2
    1706:	c5 fd 7f 15 92 57 00 	vmovdqa %ymm2,0x5792(%rip)        # 6ea0 <_ZL4MASK+0xb00>
    170d:	00 
    170e:	c4 e3 7d 02 d1 59    	vpblendd $0x59,%ymm1,%ymm0,%ymm2
    1714:	c5 fd 7f 15 a4 57 00 	vmovdqa %ymm2,0x57a4(%rip)        # 6ec0 <_ZL4MASK+0xb20>
    171b:	00 
    171c:	c4 e3 7d 02 d1 5a    	vpblendd $0x5a,%ymm1,%ymm0,%ymm2
    1722:	c5 fd 7f 15 b6 57 00 	vmovdqa %ymm2,0x57b6(%rip)        # 6ee0 <_ZL4MASK+0xb40>
    1729:	00 
    172a:	c4 e3 7d 02 d1 5b    	vpblendd $0x5b,%ymm1,%ymm0,%ymm2
    1730:	c5 fd 7f 15 c8 57 00 	vmovdqa %ymm2,0x57c8(%rip)        # 6f00 <_ZL4MASK+0xb60>
    1737:	00 
    1738:	c4 e3 7d 02 d1 5c    	vpblendd $0x5c,%ymm1,%ymm0,%ymm2
    173e:	c5 fd 7f 15 da 57 00 	vmovdqa %ymm2,0x57da(%rip)        # 6f20 <_ZL4MASK+0xb80>
    1745:	00 
    1746:	c4 e3 7d 02 d1 5d    	vpblendd $0x5d,%ymm1,%ymm0,%ymm2
    174c:	c5 fd 7f 15 ec 57 00 	vmovdqa %ymm2,0x57ec(%rip)        # 6f40 <_ZL4MASK+0xba0>
    1753:	00 
    1754:	c4 e3 7d 02 d1 5e    	vpblendd $0x5e,%ymm1,%ymm0,%ymm2
    175a:	c5 fd 7f 15 fe 57 00 	vmovdqa %ymm2,0x57fe(%rip)        # 6f60 <_ZL4MASK+0xbc0>
    1761:	00 
    1762:	c4 e3 7d 02 d1 5f    	vpblendd $0x5f,%ymm1,%ymm0,%ymm2
    1768:	c5 fd 7f 15 10 58 00 	vmovdqa %ymm2,0x5810(%rip)        # 6f80 <_ZL4MASK+0xbe0>
    176f:	00 
    1770:	c4 e3 7d 02 d1 60    	vpblendd $0x60,%ymm1,%ymm0,%ymm2
    1776:	c5 fd 7f 15 22 58 00 	vmovdqa %ymm2,0x5822(%rip)        # 6fa0 <_ZL4MASK+0xc00>
    177d:	00 
    177e:	c4 e3 7d 02 d1 61    	vpblendd $0x61,%ymm1,%ymm0,%ymm2
    1784:	c5 fd 7f 15 34 58 00 	vmovdqa %ymm2,0x5834(%rip)        # 6fc0 <_ZL4MASK+0xc20>
    178b:	00 
    178c:	c4 e3 7d 02 d1 62    	vpblendd $0x62,%ymm1,%ymm0,%ymm2
    1792:	c5 fd 7f 15 46 58 00 	vmovdqa %ymm2,0x5846(%rip)        # 6fe0 <_ZL4MASK+0xc40>
    1799:	00 
    179a:	c4 e3 7d 02 d1 63    	vpblendd $0x63,%ymm1,%ymm0,%ymm2
    17a0:	c5 fd 7f 15 58 58 00 	vmovdqa %ymm2,0x5858(%rip)        # 7000 <_ZL4MASK+0xc60>
    17a7:	00 
    17a8:	c4 e3 7d 02 d1 64    	vpblendd $0x64,%ymm1,%ymm0,%ymm2
    17ae:	c5 fd 7f 15 6a 58 00 	vmovdqa %ymm2,0x586a(%rip)        # 7020 <_ZL4MASK+0xc80>
    17b5:	00 
    17b6:	c4 e3 7d 02 d1 65    	vpblendd $0x65,%ymm1,%ymm0,%ymm2
    17bc:	c5 fd 7f 15 7c 58 00 	vmovdqa %ymm2,0x587c(%rip)        # 7040 <_ZL4MASK+0xca0>
    17c3:	00 
    17c4:	c4 e3 7d 02 d1 66    	vpblendd $0x66,%ymm1,%ymm0,%ymm2
    17ca:	c5 fd 7f 15 8e 58 00 	vmovdqa %ymm2,0x588e(%rip)        # 7060 <_ZL4MASK+0xcc0>
    17d1:	00 
    17d2:	c4 e3 7d 02 d1 67    	vpblendd $0x67,%ymm1,%ymm0,%ymm2
    17d8:	c5 fd 7f 15 a0 58 00 	vmovdqa %ymm2,0x58a0(%rip)        # 7080 <_ZL4MASK+0xce0>
    17df:	00 
    17e0:	c4 e3 7d 02 d1 68    	vpblendd $0x68,%ymm1,%ymm0,%ymm2
    17e6:	c5 fd 7f 15 b2 58 00 	vmovdqa %ymm2,0x58b2(%rip)        # 70a0 <_ZL4MASK+0xd00>
    17ed:	00 
    17ee:	c4 e3 7d 02 d1 69    	vpblendd $0x69,%ymm1,%ymm0,%ymm2
    17f4:	c5 fd 7f 15 c4 58 00 	vmovdqa %ymm2,0x58c4(%rip)        # 70c0 <_ZL4MASK+0xd20>
    17fb:	00 
    17fc:	c4 e3 7d 02 d1 6a    	vpblendd $0x6a,%ymm1,%ymm0,%ymm2
    1802:	c5 fd 7f 15 d6 58 00 	vmovdqa %ymm2,0x58d6(%rip)        # 70e0 <_ZL4MASK+0xd40>
    1809:	00 
    180a:	c4 e3 7d 02 d1 6b    	vpblendd $0x6b,%ymm1,%ymm0,%ymm2
    1810:	c5 fd 7f 15 e8 58 00 	vmovdqa %ymm2,0x58e8(%rip)        # 7100 <_ZL4MASK+0xd60>
    1817:	00 
    1818:	c4 e3 7d 02 d1 6c    	vpblendd $0x6c,%ymm1,%ymm0,%ymm2
    181e:	c5 fd 7f 15 fa 58 00 	vmovdqa %ymm2,0x58fa(%rip)        # 7120 <_ZL4MASK+0xd80>
    1825:	00 
    1826:	c4 e3 7d 02 d1 6d    	vpblendd $0x6d,%ymm1,%ymm0,%ymm2
    182c:	c5 fd 7f 15 0c 59 00 	vmovdqa %ymm2,0x590c(%rip)        # 7140 <_ZL4MASK+0xda0>
    1833:	00 
    1834:	c4 e3 7d 02 d1 6e    	vpblendd $0x6e,%ymm1,%ymm0,%ymm2
    183a:	c5 fd 7f 15 1e 59 00 	vmovdqa %ymm2,0x591e(%rip)        # 7160 <_ZL4MASK+0xdc0>
    1841:	00 
    1842:	c4 e3 7d 02 d1 6f    	vpblendd $0x6f,%ymm1,%ymm0,%ymm2
    1848:	c5 fd 7f 15 30 59 00 	vmovdqa %ymm2,0x5930(%rip)        # 7180 <_ZL4MASK+0xde0>
    184f:	00 
    1850:	c4 e3 7d 02 d1 70    	vpblendd $0x70,%ymm1,%ymm0,%ymm2
    1856:	c5 fd 7f 15 42 59 00 	vmovdqa %ymm2,0x5942(%rip)        # 71a0 <_ZL4MASK+0xe00>
    185d:	00 
    185e:	c4 e3 7d 02 d1 71    	vpblendd $0x71,%ymm1,%ymm0,%ymm2
    1864:	c5 fd 7f 15 54 59 00 	vmovdqa %ymm2,0x5954(%rip)        # 71c0 <_ZL4MASK+0xe20>
    186b:	00 
    186c:	c4 e3 7d 02 d1 72    	vpblendd $0x72,%ymm1,%ymm0,%ymm2
    1872:	c5 fd 7f 15 66 59 00 	vmovdqa %ymm2,0x5966(%rip)        # 71e0 <_ZL4MASK+0xe40>
    1879:	00 
    187a:	c4 e3 7d 02 d1 73    	vpblendd $0x73,%ymm1,%ymm0,%ymm2
    1880:	c5 fd 7f 15 78 59 00 	vmovdqa %ymm2,0x5978(%rip)        # 7200 <_ZL4MASK+0xe60>
    1887:	00 
    1888:	c4 e3 7d 02 d1 74    	vpblendd $0x74,%ymm1,%ymm0,%ymm2
    188e:	c5 fd 7f 15 8a 59 00 	vmovdqa %ymm2,0x598a(%rip)        # 7220 <_ZL4MASK+0xe80>
    1895:	00 
    1896:	c4 e3 7d 02 d1 75    	vpblendd $0x75,%ymm1,%ymm0,%ymm2
    189c:	c5 fd 7f 15 9c 59 00 	vmovdqa %ymm2,0x599c(%rip)        # 7240 <_ZL4MASK+0xea0>
    18a3:	00 
    18a4:	c4 e3 7d 02 d1 76    	vpblendd $0x76,%ymm1,%ymm0,%ymm2
    18aa:	c5 fd 7f 15 ae 59 00 	vmovdqa %ymm2,0x59ae(%rip)        # 7260 <_ZL4MASK+0xec0>
    18b1:	00 
    18b2:	c4 e3 7d 02 d1 77    	vpblendd $0x77,%ymm1,%ymm0,%ymm2
    18b8:	c5 fd 7f 15 c0 59 00 	vmovdqa %ymm2,0x59c0(%rip)        # 7280 <_ZL4MASK+0xee0>
    18bf:	00 
    18c0:	c4 e3 7d 02 d1 78    	vpblendd $0x78,%ymm1,%ymm0,%ymm2
    18c6:	c5 fd 7f 15 d2 59 00 	vmovdqa %ymm2,0x59d2(%rip)        # 72a0 <_ZL4MASK+0xf00>
    18cd:	00 
    18ce:	c4 e3 7d 02 d1 79    	vpblendd $0x79,%ymm1,%ymm0,%ymm2
    18d4:	c5 fd 7f 15 e4 59 00 	vmovdqa %ymm2,0x59e4(%rip)        # 72c0 <_ZL4MASK+0xf20>
    18db:	00 
    18dc:	c4 e3 7d 02 d1 7a    	vpblendd $0x7a,%ymm1,%ymm0,%ymm2
    18e2:	c5 fd 7f 15 f6 59 00 	vmovdqa %ymm2,0x59f6(%rip)        # 72e0 <_ZL4MASK+0xf40>
    18e9:	00 
    18ea:	c4 e3 7d 02 d1 7b    	vpblendd $0x7b,%ymm1,%ymm0,%ymm2
    18f0:	c5 fd 7f 15 08 5a 00 	vmovdqa %ymm2,0x5a08(%rip)        # 7300 <_ZL4MASK+0xf60>
    18f7:	00 
    18f8:	c4 e3 7d 02 d1 7c    	vpblendd $0x7c,%ymm1,%ymm0,%ymm2
    18fe:	c5 fd 7f 15 1a 5a 00 	vmovdqa %ymm2,0x5a1a(%rip)        # 7320 <_ZL4MASK+0xf80>
    1905:	00 
    1906:	c4 e3 7d 02 d1 7d    	vpblendd $0x7d,%ymm1,%ymm0,%ymm2
    190c:	c5 fd 7f 15 2c 5a 00 	vmovdqa %ymm2,0x5a2c(%rip)        # 7340 <_ZL4MASK+0xfa0>
    1913:	00 
    1914:	c4 e3 7d 02 d1 7e    	vpblendd $0x7e,%ymm1,%ymm0,%ymm2
    191a:	c5 fd 7f 15 3e 5a 00 	vmovdqa %ymm2,0x5a3e(%rip)        # 7360 <_ZL4MASK+0xfc0>
    1921:	00 
    1922:	c4 e3 7d 02 d1 7f    	vpblendd $0x7f,%ymm1,%ymm0,%ymm2
    1928:	c5 fd 7f 15 50 5a 00 	vmovdqa %ymm2,0x5a50(%rip)        # 7380 <_ZL4MASK+0xfe0>
    192f:	00 
    1930:	c4 e3 7d 02 d1 80    	vpblendd $0x80,%ymm1,%ymm0,%ymm2
    1936:	c5 fd 7f 15 62 5a 00 	vmovdqa %ymm2,0x5a62(%rip)        # 73a0 <_ZL4MASK+0x1000>
    193d:	00 
    193e:	c4 e3 7d 02 d1 81    	vpblendd $0x81,%ymm1,%ymm0,%ymm2
    1944:	c5 fd 7f 15 74 5a 00 	vmovdqa %ymm2,0x5a74(%rip)        # 73c0 <_ZL4MASK+0x1020>
    194b:	00 
    194c:	c4 e3 7d 02 d1 82    	vpblendd $0x82,%ymm1,%ymm0,%ymm2
    1952:	c5 fd 7f 15 86 5a 00 	vmovdqa %ymm2,0x5a86(%rip)        # 73e0 <_ZL4MASK+0x1040>
    1959:	00 
    195a:	c4 e3 7d 02 d1 83    	vpblendd $0x83,%ymm1,%ymm0,%ymm2
    1960:	c5 fd 7f 15 98 5a 00 	vmovdqa %ymm2,0x5a98(%rip)        # 7400 <_ZL4MASK+0x1060>
    1967:	00 
    1968:	c4 e3 7d 02 d1 84    	vpblendd $0x84,%ymm1,%ymm0,%ymm2
    196e:	c5 fd 7f 15 aa 5a 00 	vmovdqa %ymm2,0x5aaa(%rip)        # 7420 <_ZL4MASK+0x1080>
    1975:	00 
    1976:	c4 e3 7d 02 d1 85    	vpblendd $0x85,%ymm1,%ymm0,%ymm2
    197c:	c5 fd 7f 15 bc 5a 00 	vmovdqa %ymm2,0x5abc(%rip)        # 7440 <_ZL4MASK+0x10a0>
    1983:	00 
    1984:	c4 e3 7d 02 d1 86    	vpblendd $0x86,%ymm1,%ymm0,%ymm2
    198a:	c5 fd 7f 15 ce 5a 00 	vmovdqa %ymm2,0x5ace(%rip)        # 7460 <_ZL4MASK+0x10c0>
    1991:	00 
    1992:	c4 e3 7d 02 d1 87    	vpblendd $0x87,%ymm1,%ymm0,%ymm2
    1998:	c5 fd 7f 15 e0 5a 00 	vmovdqa %ymm2,0x5ae0(%rip)        # 7480 <_ZL4MASK+0x10e0>
    199f:	00 
    19a0:	c4 e3 7d 02 d1 88    	vpblendd $0x88,%ymm1,%ymm0,%ymm2
    19a6:	c5 fd 7f 15 f2 5a 00 	vmovdqa %ymm2,0x5af2(%rip)        # 74a0 <_ZL4MASK+0x1100>
    19ad:	00 
    19ae:	c4 e3 7d 02 d1 89    	vpblendd $0x89,%ymm1,%ymm0,%ymm2
    19b4:	c5 fd 7f 15 04 5b 00 	vmovdqa %ymm2,0x5b04(%rip)        # 74c0 <_ZL4MASK+0x1120>
    19bb:	00 
    19bc:	c4 e3 7d 02 d1 8a    	vpblendd $0x8a,%ymm1,%ymm0,%ymm2
    19c2:	c5 fd 7f 15 16 5b 00 	vmovdqa %ymm2,0x5b16(%rip)        # 74e0 <_ZL4MASK+0x1140>
    19c9:	00 
    19ca:	c4 e3 7d 02 d1 8b    	vpblendd $0x8b,%ymm1,%ymm0,%ymm2
    19d0:	c5 fd 7f 15 28 5b 00 	vmovdqa %ymm2,0x5b28(%rip)        # 7500 <_ZL4MASK+0x1160>
    19d7:	00 
    19d8:	c4 e3 7d 02 d1 8c    	vpblendd $0x8c,%ymm1,%ymm0,%ymm2
    19de:	c5 fd 7f 15 3a 5b 00 	vmovdqa %ymm2,0x5b3a(%rip)        # 7520 <_ZL4MASK+0x1180>
    19e5:	00 
    19e6:	c4 e3 7d 02 d1 8d    	vpblendd $0x8d,%ymm1,%ymm0,%ymm2
    19ec:	c5 fd 7f 15 4c 5b 00 	vmovdqa %ymm2,0x5b4c(%rip)        # 7540 <_ZL4MASK+0x11a0>
    19f3:	00 
    19f4:	c4 e3 7d 02 d1 8e    	vpblendd $0x8e,%ymm1,%ymm0,%ymm2
    19fa:	c5 fd 7f 15 5e 5b 00 	vmovdqa %ymm2,0x5b5e(%rip)        # 7560 <_ZL4MASK+0x11c0>
    1a01:	00 
    1a02:	c4 e3 7d 02 d1 8f    	vpblendd $0x8f,%ymm1,%ymm0,%ymm2
    1a08:	c5 fd 7f 15 70 5b 00 	vmovdqa %ymm2,0x5b70(%rip)        # 7580 <_ZL4MASK+0x11e0>
    1a0f:	00 
    1a10:	c4 e3 7d 02 d1 90    	vpblendd $0x90,%ymm1,%ymm0,%ymm2
    1a16:	c5 fd 7f 15 82 5b 00 	vmovdqa %ymm2,0x5b82(%rip)        # 75a0 <_ZL4MASK+0x1200>
    1a1d:	00 
    1a1e:	c4 e3 7d 02 d1 91    	vpblendd $0x91,%ymm1,%ymm0,%ymm2
    1a24:	c5 fd 7f 15 94 5b 00 	vmovdqa %ymm2,0x5b94(%rip)        # 75c0 <_ZL4MASK+0x1220>
    1a2b:	00 
    1a2c:	c4 e3 7d 02 d1 92    	vpblendd $0x92,%ymm1,%ymm0,%ymm2
    1a32:	c5 fd 7f 15 a6 5b 00 	vmovdqa %ymm2,0x5ba6(%rip)        # 75e0 <_ZL4MASK+0x1240>
    1a39:	00 
    1a3a:	c4 e3 7d 02 d1 93    	vpblendd $0x93,%ymm1,%ymm0,%ymm2
    1a40:	c5 fd 7f 15 b8 5b 00 	vmovdqa %ymm2,0x5bb8(%rip)        # 7600 <_ZL4MASK+0x1260>
    1a47:	00 
    1a48:	c4 e3 7d 02 d1 94    	vpblendd $0x94,%ymm1,%ymm0,%ymm2
    1a4e:	c5 fd 7f 15 ca 5b 00 	vmovdqa %ymm2,0x5bca(%rip)        # 7620 <_ZL4MASK+0x1280>
    1a55:	00 
    1a56:	c4 e3 7d 02 d1 95    	vpblendd $0x95,%ymm1,%ymm0,%ymm2
    1a5c:	c5 fd 7f 15 dc 5b 00 	vmovdqa %ymm2,0x5bdc(%rip)        # 7640 <_ZL4MASK+0x12a0>
    1a63:	00 
    1a64:	c4 e3 7d 02 d1 96    	vpblendd $0x96,%ymm1,%ymm0,%ymm2
    1a6a:	c5 fd 7f 15 ee 5b 00 	vmovdqa %ymm2,0x5bee(%rip)        # 7660 <_ZL4MASK+0x12c0>
    1a71:	00 
    1a72:	c4 e3 7d 02 d1 97    	vpblendd $0x97,%ymm1,%ymm0,%ymm2
    1a78:	c5 fd 7f 15 00 5c 00 	vmovdqa %ymm2,0x5c00(%rip)        # 7680 <_ZL4MASK+0x12e0>
    1a7f:	00 
    1a80:	c4 e3 7d 02 d1 98    	vpblendd $0x98,%ymm1,%ymm0,%ymm2
    1a86:	c5 fd 7f 15 12 5c 00 	vmovdqa %ymm2,0x5c12(%rip)        # 76a0 <_ZL4MASK+0x1300>
    1a8d:	00 
    1a8e:	c4 e3 7d 02 d1 99    	vpblendd $0x99,%ymm1,%ymm0,%ymm2
    1a94:	c5 fd 7f 15 24 5c 00 	vmovdqa %ymm2,0x5c24(%rip)        # 76c0 <_ZL4MASK+0x1320>
    1a9b:	00 
    1a9c:	c4 e3 7d 02 d1 9a    	vpblendd $0x9a,%ymm1,%ymm0,%ymm2
    1aa2:	c5 fd 7f 15 36 5c 00 	vmovdqa %ymm2,0x5c36(%rip)        # 76e0 <_ZL4MASK+0x1340>
    1aa9:	00 
    1aaa:	c4 e3 7d 02 d1 9b    	vpblendd $0x9b,%ymm1,%ymm0,%ymm2
    1ab0:	c5 fd 7f 15 48 5c 00 	vmovdqa %ymm2,0x5c48(%rip)        # 7700 <_ZL4MASK+0x1360>
    1ab7:	00 
    1ab8:	c4 e3 7d 02 d1 9c    	vpblendd $0x9c,%ymm1,%ymm0,%ymm2
    1abe:	c5 fd 7f 15 5a 5c 00 	vmovdqa %ymm2,0x5c5a(%rip)        # 7720 <_ZL4MASK+0x1380>
    1ac5:	00 
    1ac6:	c4 e3 7d 02 d1 9d    	vpblendd $0x9d,%ymm1,%ymm0,%ymm2
    1acc:	c5 fd 7f 15 6c 5c 00 	vmovdqa %ymm2,0x5c6c(%rip)        # 7740 <_ZL4MASK+0x13a0>
    1ad3:	00 
    1ad4:	c4 e3 7d 02 d1 9e    	vpblendd $0x9e,%ymm1,%ymm0,%ymm2
    1ada:	c5 fd 7f 15 7e 5c 00 	vmovdqa %ymm2,0x5c7e(%rip)        # 7760 <_ZL4MASK+0x13c0>
    1ae1:	00 
    1ae2:	c4 e3 7d 02 d1 9f    	vpblendd $0x9f,%ymm1,%ymm0,%ymm2
    1ae8:	c5 fd 7f 15 90 5c 00 	vmovdqa %ymm2,0x5c90(%rip)        # 7780 <_ZL4MASK+0x13e0>
    1aef:	00 
    1af0:	c4 e3 7d 02 d1 a0    	vpblendd $0xa0,%ymm1,%ymm0,%ymm2
    1af6:	c5 fd 7f 15 a2 5c 00 	vmovdqa %ymm2,0x5ca2(%rip)        # 77a0 <_ZL4MASK+0x1400>
    1afd:	00 
    1afe:	c4 e3 7d 02 d1 a1    	vpblendd $0xa1,%ymm1,%ymm0,%ymm2
    1b04:	c5 fd 7f 15 b4 5c 00 	vmovdqa %ymm2,0x5cb4(%rip)        # 77c0 <_ZL4MASK+0x1420>
    1b0b:	00 
    1b0c:	c4 e3 7d 02 d1 a2    	vpblendd $0xa2,%ymm1,%ymm0,%ymm2
    1b12:	c5 fd 7f 15 c6 5c 00 	vmovdqa %ymm2,0x5cc6(%rip)        # 77e0 <_ZL4MASK+0x1440>
    1b19:	00 
    1b1a:	c4 e3 7d 02 d1 a3    	vpblendd $0xa3,%ymm1,%ymm0,%ymm2
    1b20:	c5 fd 7f 15 d8 5c 00 	vmovdqa %ymm2,0x5cd8(%rip)        # 7800 <_ZL4MASK+0x1460>
    1b27:	00 
    1b28:	c4 e3 7d 02 d1 a4    	vpblendd $0xa4,%ymm1,%ymm0,%ymm2
    1b2e:	c5 fd 7f 15 ea 5c 00 	vmovdqa %ymm2,0x5cea(%rip)        # 7820 <_ZL4MASK+0x1480>
    1b35:	00 
    1b36:	c4 e3 7d 02 d1 a5    	vpblendd $0xa5,%ymm1,%ymm0,%ymm2
    1b3c:	c5 fd 7f 15 fc 5c 00 	vmovdqa %ymm2,0x5cfc(%rip)        # 7840 <_ZL4MASK+0x14a0>
    1b43:	00 
    1b44:	c4 e3 7d 02 d1 a6    	vpblendd $0xa6,%ymm1,%ymm0,%ymm2
    1b4a:	c5 fd 7f 15 0e 5d 00 	vmovdqa %ymm2,0x5d0e(%rip)        # 7860 <_ZL4MASK+0x14c0>
    1b51:	00 
    1b52:	c4 e3 7d 02 d1 a7    	vpblendd $0xa7,%ymm1,%ymm0,%ymm2
    1b58:	c5 fd 7f 15 20 5d 00 	vmovdqa %ymm2,0x5d20(%rip)        # 7880 <_ZL4MASK+0x14e0>
    1b5f:	00 
    1b60:	c4 e3 7d 02 d1 a8    	vpblendd $0xa8,%ymm1,%ymm0,%ymm2
    1b66:	c5 fd 7f 15 32 5d 00 	vmovdqa %ymm2,0x5d32(%rip)        # 78a0 <_ZL4MASK+0x1500>
    1b6d:	00 
    1b6e:	c4 e3 7d 02 d1 a9    	vpblendd $0xa9,%ymm1,%ymm0,%ymm2
    1b74:	c5 fd 7f 15 44 5d 00 	vmovdqa %ymm2,0x5d44(%rip)        # 78c0 <_ZL4MASK+0x1520>
    1b7b:	00 
    1b7c:	c4 e3 7d 02 d1 aa    	vpblendd $0xaa,%ymm1,%ymm0,%ymm2
    1b82:	c5 fd 7f 15 56 5d 00 	vmovdqa %ymm2,0x5d56(%rip)        # 78e0 <_ZL4MASK+0x1540>
    1b89:	00 
    1b8a:	c4 e3 7d 02 d1 ab    	vpblendd $0xab,%ymm1,%ymm0,%ymm2
    1b90:	c5 fd 7f 15 68 5d 00 	vmovdqa %ymm2,0x5d68(%rip)        # 7900 <_ZL4MASK+0x1560>
    1b97:	00 
    1b98:	c4 e3 7d 02 d1 ac    	vpblendd $0xac,%ymm1,%ymm0,%ymm2
    1b9e:	c5 fd 7f 15 7a 5d 00 	vmovdqa %ymm2,0x5d7a(%rip)        # 7920 <_ZL4MASK+0x1580>
    1ba5:	00 
    1ba6:	c4 e3 7d 02 d1 ad    	vpblendd $0xad,%ymm1,%ymm0,%ymm2
    1bac:	c5 fd 7f 15 8c 5d 00 	vmovdqa %ymm2,0x5d8c(%rip)        # 7940 <_ZL4MASK+0x15a0>
    1bb3:	00 
    1bb4:	c4 e3 7d 02 d1 ae    	vpblendd $0xae,%ymm1,%ymm0,%ymm2
    1bba:	c5 fd 7f 15 9e 5d 00 	vmovdqa %ymm2,0x5d9e(%rip)        # 7960 <_ZL4MASK+0x15c0>
    1bc1:	00 
    1bc2:	c4 e3 7d 02 d1 af    	vpblendd $0xaf,%ymm1,%ymm0,%ymm2
    1bc8:	c5 fd 7f 15 b0 5d 00 	vmovdqa %ymm2,0x5db0(%rip)        # 7980 <_ZL4MASK+0x15e0>
    1bcf:	00 
    1bd0:	c4 e3 7d 02 d1 b0    	vpblendd $0xb0,%ymm1,%ymm0,%ymm2
    1bd6:	c5 fd 7f 15 c2 5d 00 	vmovdqa %ymm2,0x5dc2(%rip)        # 79a0 <_ZL4MASK+0x1600>
    1bdd:	00 
    1bde:	c4 e3 7d 02 d1 b1    	vpblendd $0xb1,%ymm1,%ymm0,%ymm2
    1be4:	c5 fd 7f 15 d4 5d 00 	vmovdqa %ymm2,0x5dd4(%rip)        # 79c0 <_ZL4MASK+0x1620>
    1beb:	00 
    1bec:	c4 e3 7d 02 d1 b2    	vpblendd $0xb2,%ymm1,%ymm0,%ymm2
    1bf2:	c5 fd 7f 15 e6 5d 00 	vmovdqa %ymm2,0x5de6(%rip)        # 79e0 <_ZL4MASK+0x1640>
    1bf9:	00 
    1bfa:	c4 e3 7d 02 d1 b3    	vpblendd $0xb3,%ymm1,%ymm0,%ymm2
    1c00:	c5 fd 7f 15 f8 5d 00 	vmovdqa %ymm2,0x5df8(%rip)        # 7a00 <_ZL4MASK+0x1660>
    1c07:	00 
    1c08:	c4 e3 7d 02 d1 b4    	vpblendd $0xb4,%ymm1,%ymm0,%ymm2
    1c0e:	c5 fd 7f 15 0a 5e 00 	vmovdqa %ymm2,0x5e0a(%rip)        # 7a20 <_ZL4MASK+0x1680>
    1c15:	00 
    1c16:	c4 e3 7d 02 d1 b5    	vpblendd $0xb5,%ymm1,%ymm0,%ymm2
    1c1c:	c5 fd 7f 15 1c 5e 00 	vmovdqa %ymm2,0x5e1c(%rip)        # 7a40 <_ZL4MASK+0x16a0>
    1c23:	00 
    1c24:	c4 e3 7d 02 d1 b6    	vpblendd $0xb6,%ymm1,%ymm0,%ymm2
    1c2a:	c5 fd 7f 15 2e 5e 00 	vmovdqa %ymm2,0x5e2e(%rip)        # 7a60 <_ZL4MASK+0x16c0>
    1c31:	00 
    1c32:	c4 e3 7d 02 d1 b7    	vpblendd $0xb7,%ymm1,%ymm0,%ymm2
    1c38:	c5 fd 7f 15 40 5e 00 	vmovdqa %ymm2,0x5e40(%rip)        # 7a80 <_ZL4MASK+0x16e0>
    1c3f:	00 
    1c40:	c4 e3 7d 02 d1 b8    	vpblendd $0xb8,%ymm1,%ymm0,%ymm2
    1c46:	c5 fd 7f 15 52 5e 00 	vmovdqa %ymm2,0x5e52(%rip)        # 7aa0 <_ZL4MASK+0x1700>
    1c4d:	00 
    1c4e:	c4 e3 7d 02 d1 b9    	vpblendd $0xb9,%ymm1,%ymm0,%ymm2
    1c54:	c5 fd 7f 15 64 5e 00 	vmovdqa %ymm2,0x5e64(%rip)        # 7ac0 <_ZL4MASK+0x1720>
    1c5b:	00 
    1c5c:	c4 e3 7d 02 d1 ba    	vpblendd $0xba,%ymm1,%ymm0,%ymm2
    1c62:	c5 fd 7f 15 76 5e 00 	vmovdqa %ymm2,0x5e76(%rip)        # 7ae0 <_ZL4MASK+0x1740>
    1c69:	00 
    1c6a:	c4 e3 7d 02 d1 bb    	vpblendd $0xbb,%ymm1,%ymm0,%ymm2
    1c70:	c5 fd 7f 15 88 5e 00 	vmovdqa %ymm2,0x5e88(%rip)        # 7b00 <_ZL4MASK+0x1760>
    1c77:	00 
    1c78:	c4 e3 7d 02 d1 bc    	vpblendd $0xbc,%ymm1,%ymm0,%ymm2
    1c7e:	c5 fd 7f 15 9a 5e 00 	vmovdqa %ymm2,0x5e9a(%rip)        # 7b20 <_ZL4MASK+0x1780>
    1c85:	00 
    1c86:	c4 e3 7d 02 d1 bd    	vpblendd $0xbd,%ymm1,%ymm0,%ymm2
    1c8c:	c5 fd 7f 15 ac 5e 00 	vmovdqa %ymm2,0x5eac(%rip)        # 7b40 <_ZL4MASK+0x17a0>
    1c93:	00 
    1c94:	c4 e3 7d 02 d1 be    	vpblendd $0xbe,%ymm1,%ymm0,%ymm2
    1c9a:	c5 fd 7f 15 be 5e 00 	vmovdqa %ymm2,0x5ebe(%rip)        # 7b60 <_ZL4MASK+0x17c0>
    1ca1:	00 
    1ca2:	c4 e3 7d 02 d1 bf    	vpblendd $0xbf,%ymm1,%ymm0,%ymm2
    1ca8:	c5 fd 7f 15 d0 5e 00 	vmovdqa %ymm2,0x5ed0(%rip)        # 7b80 <_ZL4MASK+0x17e0>
    1caf:	00 
    1cb0:	c4 e3 7d 02 d1 c0    	vpblendd $0xc0,%ymm1,%ymm0,%ymm2
    1cb6:	c5 fd 7f 15 e2 5e 00 	vmovdqa %ymm2,0x5ee2(%rip)        # 7ba0 <_ZL4MASK+0x1800>
    1cbd:	00 
    1cbe:	c4 e3 7d 02 d1 c1    	vpblendd $0xc1,%ymm1,%ymm0,%ymm2
    1cc4:	c5 fd 7f 15 f4 5e 00 	vmovdqa %ymm2,0x5ef4(%rip)        # 7bc0 <_ZL4MASK+0x1820>
    1ccb:	00 
    1ccc:	c4 e3 7d 02 d1 c2    	vpblendd $0xc2,%ymm1,%ymm0,%ymm2
    1cd2:	c5 fd 7f 15 06 5f 00 	vmovdqa %ymm2,0x5f06(%rip)        # 7be0 <_ZL4MASK+0x1840>
    1cd9:	00 
    1cda:	c4 e3 7d 02 d1 c3    	vpblendd $0xc3,%ymm1,%ymm0,%ymm2
    1ce0:	c5 fd 7f 15 18 5f 00 	vmovdqa %ymm2,0x5f18(%rip)        # 7c00 <_ZL4MASK+0x1860>
    1ce7:	00 
    1ce8:	c4 e3 7d 02 d1 c4    	vpblendd $0xc4,%ymm1,%ymm0,%ymm2
    1cee:	c5 fd 7f 15 2a 5f 00 	vmovdqa %ymm2,0x5f2a(%rip)        # 7c20 <_ZL4MASK+0x1880>
    1cf5:	00 
    1cf6:	c4 e3 7d 02 d1 c5    	vpblendd $0xc5,%ymm1,%ymm0,%ymm2
    1cfc:	c5 fd 7f 15 3c 5f 00 	vmovdqa %ymm2,0x5f3c(%rip)        # 7c40 <_ZL4MASK+0x18a0>
    1d03:	00 
    1d04:	c4 e3 7d 02 d1 c6    	vpblendd $0xc6,%ymm1,%ymm0,%ymm2
    1d0a:	c5 fd 7f 15 4e 5f 00 	vmovdqa %ymm2,0x5f4e(%rip)        # 7c60 <_ZL4MASK+0x18c0>
    1d11:	00 
    1d12:	c4 e3 7d 02 d1 c7    	vpblendd $0xc7,%ymm1,%ymm0,%ymm2
    1d18:	c5 fd 7f 15 60 5f 00 	vmovdqa %ymm2,0x5f60(%rip)        # 7c80 <_ZL4MASK+0x18e0>
    1d1f:	00 
    1d20:	c4 e3 7d 02 d1 c8    	vpblendd $0xc8,%ymm1,%ymm0,%ymm2
    1d26:	c5 fd 7f 15 72 5f 00 	vmovdqa %ymm2,0x5f72(%rip)        # 7ca0 <_ZL4MASK+0x1900>
    1d2d:	00 
    1d2e:	c4 e3 7d 02 d1 c9    	vpblendd $0xc9,%ymm1,%ymm0,%ymm2
    1d34:	c5 fd 7f 15 84 5f 00 	vmovdqa %ymm2,0x5f84(%rip)        # 7cc0 <_ZL4MASK+0x1920>
    1d3b:	00 
    1d3c:	c4 e3 7d 02 d1 ca    	vpblendd $0xca,%ymm1,%ymm0,%ymm2
    1d42:	c5 fd 7f 15 96 5f 00 	vmovdqa %ymm2,0x5f96(%rip)        # 7ce0 <_ZL4MASK+0x1940>
    1d49:	00 
    1d4a:	c4 e3 7d 02 d1 cb    	vpblendd $0xcb,%ymm1,%ymm0,%ymm2
    1d50:	c5 fd 7f 15 a8 5f 00 	vmovdqa %ymm2,0x5fa8(%rip)        # 7d00 <_ZL4MASK+0x1960>
    1d57:	00 
    1d58:	c4 e3 7d 02 d1 cc    	vpblendd $0xcc,%ymm1,%ymm0,%ymm2
    1d5e:	c5 fd 7f 15 ba 5f 00 	vmovdqa %ymm2,0x5fba(%rip)        # 7d20 <_ZL4MASK+0x1980>
    1d65:	00 
    1d66:	c4 e3 7d 02 d1 cd    	vpblendd $0xcd,%ymm1,%ymm0,%ymm2
    1d6c:	c5 fd 7f 15 cc 5f 00 	vmovdqa %ymm2,0x5fcc(%rip)        # 7d40 <_ZL4MASK+0x19a0>
    1d73:	00 
    1d74:	c4 e3 7d 02 d1 ce    	vpblendd $0xce,%ymm1,%ymm0,%ymm2
    1d7a:	c5 fd 7f 15 de 5f 00 	vmovdqa %ymm2,0x5fde(%rip)        # 7d60 <_ZL4MASK+0x19c0>
    1d81:	00 
    1d82:	c4 e3 7d 02 d1 cf    	vpblendd $0xcf,%ymm1,%ymm0,%ymm2
    1d88:	c5 fd 7f 15 f0 5f 00 	vmovdqa %ymm2,0x5ff0(%rip)        # 7d80 <_ZL4MASK+0x19e0>
    1d8f:	00 
    1d90:	c4 e3 7d 02 d1 d0    	vpblendd $0xd0,%ymm1,%ymm0,%ymm2
    1d96:	c5 fd 7f 15 02 60 00 	vmovdqa %ymm2,0x6002(%rip)        # 7da0 <_ZL4MASK+0x1a00>
    1d9d:	00 
    1d9e:	c4 e3 7d 02 d1 d1    	vpblendd $0xd1,%ymm1,%ymm0,%ymm2
    1da4:	c5 fd 7f 15 14 60 00 	vmovdqa %ymm2,0x6014(%rip)        # 7dc0 <_ZL4MASK+0x1a20>
    1dab:	00 
    1dac:	c4 e3 7d 02 d1 d2    	vpblendd $0xd2,%ymm1,%ymm0,%ymm2
    1db2:	c5 fd 7f 15 26 60 00 	vmovdqa %ymm2,0x6026(%rip)        # 7de0 <_ZL4MASK+0x1a40>
    1db9:	00 
    1dba:	c4 e3 7d 02 d1 d3    	vpblendd $0xd3,%ymm1,%ymm0,%ymm2
    1dc0:	c5 fd 7f 15 38 60 00 	vmovdqa %ymm2,0x6038(%rip)        # 7e00 <_ZL4MASK+0x1a60>
    1dc7:	00 
    1dc8:	c4 e3 7d 02 d1 d4    	vpblendd $0xd4,%ymm1,%ymm0,%ymm2
    1dce:	c5 fd 7f 15 4a 60 00 	vmovdqa %ymm2,0x604a(%rip)        # 7e20 <_ZL4MASK+0x1a80>
    1dd5:	00 
    1dd6:	c4 e3 7d 02 d1 d5    	vpblendd $0xd5,%ymm1,%ymm0,%ymm2
    1ddc:	c5 fd 7f 15 5c 60 00 	vmovdqa %ymm2,0x605c(%rip)        # 7e40 <_ZL4MASK+0x1aa0>
    1de3:	00 
    1de4:	c4 e3 7d 02 d1 d6    	vpblendd $0xd6,%ymm1,%ymm0,%ymm2
    1dea:	c5 fd 7f 15 6e 60 00 	vmovdqa %ymm2,0x606e(%rip)        # 7e60 <_ZL4MASK+0x1ac0>
    1df1:	00 
    1df2:	c4 e3 7d 02 d1 d7    	vpblendd $0xd7,%ymm1,%ymm0,%ymm2
    1df8:	c5 fd 7f 15 80 60 00 	vmovdqa %ymm2,0x6080(%rip)        # 7e80 <_ZL4MASK+0x1ae0>
    1dff:	00 
    1e00:	c4 e3 7d 02 d1 d8    	vpblendd $0xd8,%ymm1,%ymm0,%ymm2
    1e06:	c5 fd 7f 15 92 60 00 	vmovdqa %ymm2,0x6092(%rip)        # 7ea0 <_ZL4MASK+0x1b00>
    1e0d:	00 
    1e0e:	c4 e3 7d 02 d1 d9    	vpblendd $0xd9,%ymm1,%ymm0,%ymm2
    1e14:	c5 fd 7f 15 a4 60 00 	vmovdqa %ymm2,0x60a4(%rip)        # 7ec0 <_ZL4MASK+0x1b20>
    1e1b:	00 
    1e1c:	c4 e3 7d 02 d1 da    	vpblendd $0xda,%ymm1,%ymm0,%ymm2
    1e22:	c5 fd 7f 15 b6 60 00 	vmovdqa %ymm2,0x60b6(%rip)        # 7ee0 <_ZL4MASK+0x1b40>
    1e29:	00 
    1e2a:	c4 e3 7d 02 d1 db    	vpblendd $0xdb,%ymm1,%ymm0,%ymm2
    1e30:	c5 fd 7f 15 c8 60 00 	vmovdqa %ymm2,0x60c8(%rip)        # 7f00 <_ZL4MASK+0x1b60>
    1e37:	00 
    1e38:	c4 e3 7d 02 d1 dc    	vpblendd $0xdc,%ymm1,%ymm0,%ymm2
    1e3e:	c5 fd 7f 15 da 60 00 	vmovdqa %ymm2,0x60da(%rip)        # 7f20 <_ZL4MASK+0x1b80>
    1e45:	00 
    1e46:	c4 e3 7d 02 d1 dd    	vpblendd $0xdd,%ymm1,%ymm0,%ymm2
    1e4c:	c5 fd 7f 15 ec 60 00 	vmovdqa %ymm2,0x60ec(%rip)        # 7f40 <_ZL4MASK+0x1ba0>
    1e53:	00 
    1e54:	c4 e3 7d 02 d1 de    	vpblendd $0xde,%ymm1,%ymm0,%ymm2
    1e5a:	c5 fd 7f 15 fe 60 00 	vmovdqa %ymm2,0x60fe(%rip)        # 7f60 <_ZL4MASK+0x1bc0>
    1e61:	00 
    1e62:	c4 e3 7d 02 d1 df    	vpblendd $0xdf,%ymm1,%ymm0,%ymm2
    1e68:	c5 fd 7f 15 10 61 00 	vmovdqa %ymm2,0x6110(%rip)        # 7f80 <_ZL4MASK+0x1be0>
    1e6f:	00 
    1e70:	c4 e3 7d 02 d1 e0    	vpblendd $0xe0,%ymm1,%ymm0,%ymm2
    1e76:	c5 fd 7f 15 22 61 00 	vmovdqa %ymm2,0x6122(%rip)        # 7fa0 <_ZL4MASK+0x1c00>
    1e7d:	00 
    1e7e:	c4 e3 7d 02 d1 e1    	vpblendd $0xe1,%ymm1,%ymm0,%ymm2
    1e84:	c5 fd 7f 15 34 61 00 	vmovdqa %ymm2,0x6134(%rip)        # 7fc0 <_ZL4MASK+0x1c20>
    1e8b:	00 
    1e8c:	c4 e3 7d 02 d1 e2    	vpblendd $0xe2,%ymm1,%ymm0,%ymm2
    1e92:	c5 fd 7f 15 46 61 00 	vmovdqa %ymm2,0x6146(%rip)        # 7fe0 <_ZL4MASK+0x1c40>
    1e99:	00 
    1e9a:	c4 e3 7d 02 d1 e3    	vpblendd $0xe3,%ymm1,%ymm0,%ymm2
    1ea0:	c5 fd 7f 15 58 61 00 	vmovdqa %ymm2,0x6158(%rip)        # 8000 <_ZL4MASK+0x1c60>
    1ea7:	00 
    1ea8:	c4 e3 7d 02 d1 e4    	vpblendd $0xe4,%ymm1,%ymm0,%ymm2
    1eae:	c5 fd 7f 15 6a 61 00 	vmovdqa %ymm2,0x616a(%rip)        # 8020 <_ZL4MASK+0x1c80>
    1eb5:	00 
    1eb6:	c4 e3 7d 02 d1 e5    	vpblendd $0xe5,%ymm1,%ymm0,%ymm2
    1ebc:	c5 fd 7f 15 7c 61 00 	vmovdqa %ymm2,0x617c(%rip)        # 8040 <_ZL4MASK+0x1ca0>
    1ec3:	00 
    1ec4:	c4 e3 7d 02 d1 e6    	vpblendd $0xe6,%ymm1,%ymm0,%ymm2
    1eca:	c5 fd 7f 15 8e 61 00 	vmovdqa %ymm2,0x618e(%rip)        # 8060 <_ZL4MASK+0x1cc0>
    1ed1:	00 
    1ed2:	c4 e3 7d 02 d1 e7    	vpblendd $0xe7,%ymm1,%ymm0,%ymm2
    1ed8:	c5 fd 7f 15 a0 61 00 	vmovdqa %ymm2,0x61a0(%rip)        # 8080 <_ZL4MASK+0x1ce0>
    1edf:	00 
    1ee0:	c4 e3 7d 02 d1 e8    	vpblendd $0xe8,%ymm1,%ymm0,%ymm2
    1ee6:	c5 fd 7f 15 b2 61 00 	vmovdqa %ymm2,0x61b2(%rip)        # 80a0 <_ZL4MASK+0x1d00>
    1eed:	00 
    1eee:	c4 e3 7d 02 d1 e9    	vpblendd $0xe9,%ymm1,%ymm0,%ymm2
    1ef4:	c5 fd 7f 15 c4 61 00 	vmovdqa %ymm2,0x61c4(%rip)        # 80c0 <_ZL4MASK+0x1d20>
    1efb:	00 
    1efc:	c4 e3 7d 02 d1 ea    	vpblendd $0xea,%ymm1,%ymm0,%ymm2
    1f02:	c5 fd 7f 15 d6 61 00 	vmovdqa %ymm2,0x61d6(%rip)        # 80e0 <_ZL4MASK+0x1d40>
    1f09:	00 
    1f0a:	c4 e3 7d 02 d1 eb    	vpblendd $0xeb,%ymm1,%ymm0,%ymm2
    1f10:	c5 fd 7f 15 e8 61 00 	vmovdqa %ymm2,0x61e8(%rip)        # 8100 <_ZL4MASK+0x1d60>
    1f17:	00 
    1f18:	c4 e3 7d 02 d1 ec    	vpblendd $0xec,%ymm1,%ymm0,%ymm2
    1f1e:	c5 fd 7f 15 fa 61 00 	vmovdqa %ymm2,0x61fa(%rip)        # 8120 <_ZL4MASK+0x1d80>
    1f25:	00 
    1f26:	c4 e3 7d 02 d1 ed    	vpblendd $0xed,%ymm1,%ymm0,%ymm2
    1f2c:	c5 fd 7f 15 0c 62 00 	vmovdqa %ymm2,0x620c(%rip)        # 8140 <_ZL4MASK+0x1da0>
    1f33:	00 
    1f34:	c4 e3 7d 02 d1 ee    	vpblendd $0xee,%ymm1,%ymm0,%ymm2
    1f3a:	c5 fd 7f 15 1e 62 00 	vmovdqa %ymm2,0x621e(%rip)        # 8160 <_ZL4MASK+0x1dc0>
    1f41:	00 
    1f42:	c4 e3 7d 02 d1 ef    	vpblendd $0xef,%ymm1,%ymm0,%ymm2
    1f48:	c5 fd 7f 15 30 62 00 	vmovdqa %ymm2,0x6230(%rip)        # 8180 <_ZL4MASK+0x1de0>
    1f4f:	00 
    1f50:	c4 e3 7d 02 d1 f0    	vpblendd $0xf0,%ymm1,%ymm0,%ymm2
    1f56:	c5 fd 7f 15 42 62 00 	vmovdqa %ymm2,0x6242(%rip)        # 81a0 <_ZL4MASK+0x1e00>
    1f5d:	00 
    1f5e:	c4 e3 7d 02 d1 f1    	vpblendd $0xf1,%ymm1,%ymm0,%ymm2
    1f64:	c5 fd 7f 15 54 62 00 	vmovdqa %ymm2,0x6254(%rip)        # 81c0 <_ZL4MASK+0x1e20>
    1f6b:	00 
    1f6c:	c4 e3 7d 02 d1 f2    	vpblendd $0xf2,%ymm1,%ymm0,%ymm2
    1f72:	c5 fd 7f 15 66 62 00 	vmovdqa %ymm2,0x6266(%rip)        # 81e0 <_ZL4MASK+0x1e40>
    1f79:	00 
    1f7a:	c4 e3 7d 02 d1 f3    	vpblendd $0xf3,%ymm1,%ymm0,%ymm2
    1f80:	c5 fd 7f 15 78 62 00 	vmovdqa %ymm2,0x6278(%rip)        # 8200 <_ZL4MASK+0x1e60>
    1f87:	00 
    1f88:	c4 e3 7d 02 d1 f4    	vpblendd $0xf4,%ymm1,%ymm0,%ymm2
    1f8e:	c5 fd 7f 15 8a 62 00 	vmovdqa %ymm2,0x628a(%rip)        # 8220 <_ZL4MASK+0x1e80>
    1f95:	00 
    1f96:	c4 e3 7d 02 d1 f5    	vpblendd $0xf5,%ymm1,%ymm0,%ymm2
    1f9c:	c5 fd 7f 15 9c 62 00 	vmovdqa %ymm2,0x629c(%rip)        # 8240 <_ZL4MASK+0x1ea0>
    1fa3:	00 
    1fa4:	c4 e3 7d 02 d1 f6    	vpblendd $0xf6,%ymm1,%ymm0,%ymm2
    1faa:	c5 fd 7f 15 ae 62 00 	vmovdqa %ymm2,0x62ae(%rip)        # 8260 <_ZL4MASK+0x1ec0>
    1fb1:	00 
    1fb2:	c4 e3 7d 02 d1 f7    	vpblendd $0xf7,%ymm1,%ymm0,%ymm2
    1fb8:	c5 fd 7f 15 c0 62 00 	vmovdqa %ymm2,0x62c0(%rip)        # 8280 <_ZL4MASK+0x1ee0>
    1fbf:	00 
    1fc0:	c4 e3 7d 02 d1 f8    	vpblendd $0xf8,%ymm1,%ymm0,%ymm2
    1fc6:	c5 fd 7f 15 d2 62 00 	vmovdqa %ymm2,0x62d2(%rip)        # 82a0 <_ZL4MASK+0x1f00>
    1fcd:	00 
    1fce:	c4 e3 7d 02 d1 f9    	vpblendd $0xf9,%ymm1,%ymm0,%ymm2
    1fd4:	c5 fd 7f 15 e4 62 00 	vmovdqa %ymm2,0x62e4(%rip)        # 82c0 <_ZL4MASK+0x1f20>
    1fdb:	00 
    1fdc:	c4 e3 7d 02 d1 fa    	vpblendd $0xfa,%ymm1,%ymm0,%ymm2
    1fe2:	c5 fd 7f 15 f6 62 00 	vmovdqa %ymm2,0x62f6(%rip)        # 82e0 <_ZL4MASK+0x1f40>
    1fe9:	00 
    1fea:	c4 e3 7d 02 d1 fb    	vpblendd $0xfb,%ymm1,%ymm0,%ymm2
    1ff0:	c5 fd 7f 15 08 63 00 	vmovdqa %ymm2,0x6308(%rip)        # 8300 <_ZL4MASK+0x1f60>
    1ff7:	00 
    1ff8:	c4 e3 7d 02 d1 fc    	vpblendd $0xfc,%ymm1,%ymm0,%ymm2
    1ffe:	c5 fd 7f 15 1a 63 00 	vmovdqa %ymm2,0x631a(%rip)        # 8320 <_ZL4MASK+0x1f80>
    2005:	00 
    2006:	c4 e3 7d 02 d1 fd    	vpblendd $0xfd,%ymm1,%ymm0,%ymm2
    200c:	c4 e3 7d 02 c1 fe    	vpblendd $0xfe,%ymm1,%ymm0,%ymm0
    2012:	c5 fd 7f 15 26 63 00 	vmovdqa %ymm2,0x6326(%rip)        # 8340 <_ZL4MASK+0x1fa0>
    2019:	00 
    201a:	c5 fd 7f 05 3e 63 00 	vmovdqa %ymm0,0x633e(%rip)        # 8360 <_ZL4MASK+0x1fc0>
    2021:	00 
    2022:	c5 fd 7f 0d 56 63 00 	vmovdqa %ymm1,0x6356(%rip)        # 8380 <_ZL4MASK+0x1fe0>
    2029:	00 
    202a:	c5 f8 77             	vzeroupper 
    202d:	c9                   	leaveq 
    202e:	c3                   	retq   
    202f:	90                   	nop

0000000000002030 <main>:
    2030:	55                   	push   %rbp
    2031:	31 ff                	xor    %edi,%edi
    2033:	48 89 e5             	mov    %rsp,%rbp
    2036:	41 57                	push   %r15
    2038:	41 56                	push   %r14
    203a:	41 55                	push   %r13
    203c:	41 54                	push   %r12
    203e:	53                   	push   %rbx
    203f:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    2043:	48 81 ec 00 15 00 00 	sub    $0x1500,%rsp
    204a:	e8 f1 ef ff ff       	callq  1040 <_ZNSt8ios_base15sync_with_stdioEb@plt>
    204f:	48 8d 74 24 70       	lea    0x70(%rsp),%rsi
    2054:	ba 75 6c 00 00       	mov    $0x6c75,%edx
    2059:	c7 84 24 80 00 00 00 	movl   $0x61666564,0x80(%rsp)
    2060:	64 65 66 61 
    2064:	48 8d 46 10          	lea    0x10(%rsi),%rax
    2068:	48 8d bc 24 70 01 00 	lea    0x170(%rsp),%rdi
    206f:	00 
    2070:	48 89 f3             	mov    %rsi,%rbx
    2073:	48 89 74 24 28       	mov    %rsi,0x28(%rsp)
    2078:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
    207d:	66 89 50 04          	mov    %dx,0x4(%rax)
    2081:	c6 40 06 74          	movb   $0x74,0x6(%rax)
    2085:	48 c7 05 78 32 00 00 	movq   $0x0,0x3278(%rip)        # 5308 <_ZSt3cin@@GLIBCXX_3.4+0xe8>
    208c:	00 00 00 00 
    2090:	48 c7 44 24 78 07 00 	movq   $0x7,0x78(%rsp)
    2097:	00 00 
    2099:	c6 84 24 87 00 00 00 	movb   $0x0,0x87(%rsp)
    20a0:	00 
    20a1:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
    20a6:	e8 b5 f0 ff ff       	callq  1160 <_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@plt>
    20ab:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    20b0:	48 8d 43 10          	lea    0x10(%rbx),%rax
    20b4:	48 39 c7             	cmp    %rax,%rdi
    20b7:	74 05                	je     20be <main+0x8e>
    20b9:	e8 e2 ef ff ff       	callq  10a0 <_ZdlPv@plt>
    20be:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
    20c3:	48 89 df             	mov    %rbx,%rdi
    20c6:	e8 35 f0 ff ff       	callq  1100 <_ZNSt13random_device9_M_getvalEv@plt>
    20cb:	48 89 df             	mov    %rbx,%rdi
    20ce:	e8 bd ef ff ff       	callq  1090 <_ZNSt13random_device7_M_finiEv@plt>
    20d3:	be 00 00 40 00       	mov    $0x400000,%esi
    20d8:	bf 80 00 00 00       	mov    $0x80,%edi
    20dd:	e8 8e ef ff ff       	callq  1070 <aligned_alloc@plt>
    20e2:	be 80 00 00 00       	mov    $0x80,%esi
    20e7:	bf 80 00 00 00       	mov    $0x80,%edi
    20ec:	49 89 c4             	mov    %rax,%r12
    20ef:	e8 7c ef ff ff       	callq  1070 <aligned_alloc@plt>
    20f4:	48 8d 35 5d 0f 00 00 	lea    0xf5d(%rip),%rsi        # 3058 <_IO_stdin_used+0x58>
    20fb:	48 8d 3d fe 2f 00 00 	lea    0x2ffe(%rip),%rdi        # 5100 <_ZSt4cout@@GLIBCXX_3.4>
    2102:	49 89 c7             	mov    %rax,%r15
    2105:	e8 a6 ef ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    210a:	c5 fd 6f 15 4e 10 00 	vmovdqa 0x104e(%rip),%ymm2        # 3160 <_IO_stdin_used+0x160>
    2111:	00 
    2112:	c5 fd 6f 25 66 10 00 	vmovdqa 0x1066(%rip),%ymm4        # 3180 <_IO_stdin_used+0x180>
    2119:	00 
    211a:	4c 89 e0             	mov    %r12,%rax
    211d:	c5 fd 28 1d 7b 10 00 	vmovapd 0x107b(%rip),%ymm3        # 31a0 <_IO_stdin_used+0x1a0>
    2124:	00 
    2125:	49 8d 94 24 00 00 40 	lea    0x400000(%r12),%rdx
    212c:	00 
    212d:	0f 1f 00             	nopl   (%rax)
    2130:	c5 ed fe 05 c8 0f 00 	vpaddd 0xfc8(%rip),%ymm2,%ymm0        # 3100 <_IO_stdin_used+0x100>
    2137:	00 
    2138:	c4 e3 7d 39 d5 01    	vextracti128 $0x1,%ymm2,%xmm5
    213e:	48 83 c0 20          	add    $0x20,%rax
    2142:	c5 fe e6 ed          	vcvtdq2pd %xmm5,%ymm5
    2146:	c5 d5 59 eb          	vmulpd %ymm3,%ymm5,%ymm5
    214a:	c5 fe e6 c8          	vcvtdq2pd %xmm0,%ymm1
    214e:	c5 f5 59 cb          	vmulpd %ymm3,%ymm1,%ymm1
    2152:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
    2158:	c5 fe e6 c0          	vcvtdq2pd %xmm0,%ymm0
    215c:	c5 fd 59 c3          	vmulpd %ymm3,%ymm0,%ymm0
    2160:	c5 fd e6 ed          	vcvttpd2dq %ymm5,%xmm5
    2164:	c5 fd e6 c9          	vcvttpd2dq %ymm1,%xmm1
    2168:	c5 fd e6 c0          	vcvttpd2dq %ymm0,%xmm0
    216c:	c4 e3 75 38 c0 01    	vinserti128 $0x1,%xmm0,%ymm1,%ymm0
    2172:	c5 fe e6 ca          	vcvtdq2pd %xmm2,%ymm1
    2176:	c5 ed fe d4          	vpaddd %ymm4,%ymm2,%ymm2
    217a:	c5 f5 59 cb          	vmulpd %ymm3,%ymm1,%ymm1
    217e:	c5 fd e6 c9          	vcvttpd2dq %ymm1,%xmm1
    2182:	c4 e3 75 38 cd 01    	vinserti128 $0x1,%xmm5,%ymm1,%ymm1
    2188:	c5 fd fa c1          	vpsubd %ymm1,%ymm0,%ymm0
    218c:	c5 fe 7f 40 e0       	vmovdqu %ymm0,-0x20(%rax)
    2191:	48 39 c2             	cmp    %rax,%rdx
    2194:	75 9a                	jne    2130 <main+0x100>
    2196:	48 8d 35 99 0e 00 00 	lea    0xe99(%rip),%rsi        # 3036 <_IO_stdin_used+0x36>
    219d:	48 8d 3d 5c 2f 00 00 	lea    0x2f5c(%rip),%rdi        # 5100 <_ZSt4cout@@GLIBCXX_3.4>
    21a4:	c5 f8 77             	vzeroupper 
    21a7:	e8 04 ef ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    21ac:	31 db                	xor    %ebx,%ebx
    21ae:	45 31 ed             	xor    %r13d,%r13d
    21b1:	48 8d 35 d0 0e 00 00 	lea    0xed0(%rip),%rsi        # 3088 <_IO_stdin_used+0x88>
    21b8:	48 8d 3d 41 2f 00 00 	lea    0x2f41(%rip),%rdi        # 5100 <_ZSt4cout@@GLIBCXX_3.4>
    21bf:	e8 ec ee ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    21c4:	31 c9                	xor    %ecx,%ecx
    21c6:	48 c7 44 24 38 00 00 	movq   $0x0,0x38(%rsp)
    21cd:	00 00 
    21cf:	eb 1b                	jmp    21ec <main+0x1bc>
    21d1:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    21d8:	00 00 00 00 
    21dc:	0f 1f 40 00          	nopl   0x0(%rax)
    21e0:	48 ff c3             	inc    %rbx
    21e3:	48 81 fb 00 00 10 00 	cmp    $0x100000,%rbx
    21ea:	74 4b                	je     2237 <main+0x207>
    21ec:	41 83 3c 9c 01       	cmpl   $0x1,(%r12,%rbx,4)
    21f1:	75 ed                	jne    21e0 <main+0x1b0>
    21f3:	b8 ab aa aa aa       	mov    $0xaaaaaaab,%eax
    21f8:	41 c7 04 9c 00 00 00 	movl   $0x0,(%r12,%rbx,4)
    21ff:	00 
    2200:	f7 e3                	mul    %ebx
    2202:	89 d8                	mov    %ebx,%eax
    2204:	d1 ea                	shr    %edx
    2206:	29 d0                	sub    %edx,%eax
    2208:	83 e0 01             	and    $0x1,%eax
    220b:	44 8d b4 43 ff ff 0f 	lea    0xfffff(%rbx,%rax,2),%r14d
    2212:	00 
    2213:	41 81 e6 ff ff 0f 00 	and    $0xfffff,%r14d
    221a:	4c 39 e9             	cmp    %r13,%rcx
    221d:	0f 84 7d 02 00 00    	je     24a0 <main+0x470>
    2223:	48 ff c3             	inc    %rbx
    2226:	45 89 75 00          	mov    %r14d,0x0(%r13)
    222a:	49 83 c5 04          	add    $0x4,%r13
    222e:	48 81 fb 00 00 10 00 	cmp    $0x100000,%rbx
    2235:	75 b5                	jne    21ec <main+0x1bc>
    2237:	4c 2b 6c 24 38       	sub    0x38(%rsp),%r13
    223c:	31 d2                	xor    %edx,%edx
    223e:	31 c0                	xor    %eax,%eax
    2240:	49 c1 fd 02          	sar    $0x2,%r13
    2244:	74 22                	je     2268 <main+0x238>
    2246:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    224d:	00 00 00 
    2250:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
    2255:	48 63 14 96          	movslq (%rsi,%rdx,4),%rdx
    2259:	41 ff 04 94          	incl   (%r12,%rdx,4)
    225d:	8d 50 01             	lea    0x1(%rax),%edx
    2260:	48 89 d0             	mov    %rdx,%rax
    2263:	49 39 d5             	cmp    %rdx,%r13
    2266:	77 e8                	ja     2250 <main+0x220>
    2268:	48 83 7c 24 38 00    	cmpq   $0x0,0x38(%rsp)
    226e:	74 0a                	je     227a <main+0x24a>
    2270:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    2275:	e8 26 ee ff ff       	callq  10a0 <_ZdlPv@plt>
    227a:	48 8d 35 b5 0d 00 00 	lea    0xdb5(%rip),%rsi        # 3036 <_IO_stdin_used+0x36>
    2281:	48 8d 3d 78 2e 00 00 	lea    0x2e78(%rip),%rdi        # 5100 <_ZSt4cout@@GLIBCXX_3.4>
    2288:	45 31 ed             	xor    %r13d,%r13d
    228b:	45 31 f6             	xor    %r14d,%r14d
    228e:	e8 1d ee ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    2293:	48 8d 35 26 0e 00 00 	lea    0xe26(%rip),%rsi        # 30c0 <_IO_stdin_used+0xc0>
    229a:	48 8d 3d 5f 2e 00 00 	lea    0x2e5f(%rip),%rdi        # 5100 <_ZSt4cout@@GLIBCXX_3.4>
    22a1:	e8 0a ee ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    22a6:	48 8d 3d 53 2e 00 00 	lea    0x2e53(%rip),%rdi        # 5100 <_ZSt4cout@@GLIBCXX_3.4>
    22ad:	e8 ae ed ff ff       	callq  1060 <_ZNSo5flushEv@plt>
    22b2:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    22b7:	ba 10 00 00 00       	mov    $0x10,%edx
    22bc:	48 8d 35 7a 0d 00 00 	lea    0xd7a(%rip),%rsi        # 303d <_IO_stdin_used+0x3d>
    22c3:	e8 08 ee ff ff       	callq  10d0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@plt>
    22c8:	48 8d 44 24 4c       	lea    0x4c(%rsp),%rax
    22cd:	c4 c1 f9 6e e4       	vmovq  %r12,%xmm4
    22d2:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    22d7:	c4 c3 d9 22 e7 01    	vpinsrq $0x1,%r15,%xmm4,%xmm4
    22dd:	48 8d 44 24 50       	lea    0x50(%rsp),%rax
    22e2:	c5 f8 29 64 24 10    	vmovaps %xmm4,0x10(%rsp)
    22e8:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    22ed:	e9 64 01 00 00       	jmpq   2456 <main+0x426>
    22f2:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    22f9:	00 00 00 00 
    22fd:	0f 1f 00             	nopl   (%rax)
    2300:	48 8b 7c 24 28       	mov    0x28(%rsp),%rdi
    2305:	c5 fa 7e 74 24 38    	vmovq  0x38(%rsp),%xmm6
    230b:	4c 89 f0             	mov    %r14,%rax
    230e:	31 d2                	xor    %edx,%edx
    2310:	c5 fd 6f 3d 08 0e 00 	vmovdqa 0xe08(%rip),%ymm7        # 3120 <_IO_stdin_used+0x120>
    2317:	00 
    2318:	c4 c2 45 8c 47 fc    	vpmaskmovd -0x4(%r15),%ymm7,%ymm0
    231e:	b9 20 00 00 00       	mov    $0x20,%ecx
    2323:	c4 c2 45 8c 4c 24 fc 	vpmaskmovd -0x4(%r12),%ymm7,%ymm1
    232a:	c5 fd fe c1          	vpaddd %ymm1,%ymm0,%ymm0
    232e:	c4 e3 c9 22 d7 01    	vpinsrq $0x1,%rdi,%xmm6,%xmm2
    2334:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    2339:	c4 c2 45 8e 44 24 fc 	vpmaskmovd %ymm0,%ymm7,-0x4(%r12)
    2340:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    2343:	c5 f8 29 54 24 60    	vmovaps %xmm2,0x60(%rsp)
    2349:	c5 fd 66 05 af 0d 00 	vpcmpgtd 0xdaf(%rip),%ymm0,%ymm0        # 3100 <_IO_stdin_used+0x100>
    2350:	00 
    2351:	c5 fd db 05 a7 0d 00 	vpand  0xda7(%rip),%ymm0,%ymm0        # 3100 <_IO_stdin_used+0x100>
    2358:	00 
    2359:	48 8d 3d 90 06 00 00 	lea    0x690(%rip),%rdi        # 29f0 <_Z9descargarPiS_._omp_fn.0>
    2360:	c5 fd db c7          	vpand  %ymm7,%ymm0,%ymm0
    2364:	c5 f9 6f 7c 24 10    	vmovdqa 0x10(%rsp),%xmm7
    236a:	c4 e3 7d 39 c1 01    	vextracti128 $0x1,%ymm0,%xmm1
    2370:	c5 f1 fe c0          	vpaddd %xmm0,%xmm1,%xmm0
    2374:	c5 f8 29 7c 24 50    	vmovaps %xmm7,0x50(%rsp)
    237a:	c5 f1 73 d8 08       	vpsrldq $0x8,%xmm0,%xmm1
    237f:	c5 f9 fe c1          	vpaddd %xmm1,%xmm0,%xmm0
    2383:	c5 f1 73 d8 04       	vpsrldq $0x4,%xmm0,%xmm1
    2388:	c5 f9 fe c1          	vpaddd %xmm1,%xmm0,%xmm0
    238c:	c5 f9 7e 44 24 4c    	vmovd  %xmm0,0x4c(%rsp)
    2392:	c5 f8 77             	vzeroupper 
    2395:	31 db                	xor    %ebx,%ebx
    2397:	e8 84 ed ff ff       	callq  1120 <GOMP_parallel@plt>
    239c:	41 8b 74 24 1c       	mov    0x1c(%r12),%esi
    23a1:	45 8b 44 24 20       	mov    0x20(%r12),%r8d
    23a6:	41 8b 7f 20          	mov    0x20(%r15),%edi
    23aa:	41 8b 4f 1c          	mov    0x1c(%r15),%ecx
    23ae:	41 8b 57 7c          	mov    0x7c(%r15),%edx
    23b2:	41 03 94 24 fc ff 3f 	add    0x3ffffc(%r12),%edx
    23b9:	00 
    23ba:	44 01 c7             	add    %r8d,%edi
    23bd:	01 f1                	add    %esi,%ecx
    23bf:	41 8b 07             	mov    (%r15),%eax
    23c2:	41 03 04 24          	add    (%r12),%eax
    23c6:	83 fa 01             	cmp    $0x1,%edx
    23c9:	41 89 04 24          	mov    %eax,(%r12)
    23cd:	0f 9f c3             	setg   %bl
    23d0:	83 f8 01             	cmp    $0x1,%eax
    23d3:	41 89 7c 24 20       	mov    %edi,0x20(%r12)
    23d8:	0f 9f c0             	setg   %al
    23db:	41 89 94 24 fc ff 3f 	mov    %edx,0x3ffffc(%r12)
    23e2:	00 
    23e3:	0f b6 c0             	movzbl %al,%eax
    23e6:	41 89 4c 24 1c       	mov    %ecx,0x1c(%r12)
    23eb:	01 c3                	add    %eax,%ebx
    23ed:	31 c0                	xor    %eax,%eax
    23ef:	03 5c 24 4c          	add    0x4c(%rsp),%ebx
    23f3:	41 83 f8 01          	cmp    $0x1,%r8d
    23f7:	0f 9f c0             	setg   %al
    23fa:	29 c3                	sub    %eax,%ebx
    23fc:	31 c0                	xor    %eax,%eax
    23fe:	83 ff 01             	cmp    $0x1,%edi
    2401:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    2406:	0f 9f c0             	setg   %al
    2409:	01 c3                	add    %eax,%ebx
    240b:	31 c0                	xor    %eax,%eax
    240d:	83 fe 01             	cmp    $0x1,%esi
    2410:	0f 9f c0             	setg   %al
    2413:	29 c3                	sub    %eax,%ebx
    2415:	31 c0                	xor    %eax,%eax
    2417:	83 f9 01             	cmp    $0x1,%ecx
    241a:	0f 9f c0             	setg   %al
    241d:	01 c3                	add    %eax,%ebx
    241f:	89 de                	mov    %ebx,%esi
    2421:	e8 4a ed ff ff       	callq  1170 <_ZNSolsEi@plt>
    2426:	ba 01 00 00 00       	mov    $0x1,%edx
    242b:	48 8d 35 e5 0b 00 00 	lea    0xbe5(%rip),%rsi        # 3017 <_IO_stdin_used+0x17>
    2432:	48 89 c7             	mov    %rax,%rdi
    2435:	e8 a6 ec ff ff       	callq  10e0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    243a:	41 8d 45 01          	lea    0x1(%r13),%eax
    243e:	3d 0f 27 00 00       	cmp    $0x270f,%eax
    2443:	0f 9f c2             	setg   %dl
    2446:	85 db                	test   %ebx,%ebx
    2448:	0f 9e c0             	setle  %al
    244b:	49 ff c5             	inc    %r13
    244e:	08 c2                	or     %al,%dl
    2450:	0f 85 38 01 00 00    	jne    258e <main+0x55e>
    2456:	c4 c1 7e 6f 0c 24    	vmovdqu (%r12),%ymm1
    245c:	b9 10 00 00 00       	mov    $0x10,%ecx
    2461:	4c 89 ff             	mov    %r15,%rdi
    2464:	4c 89 f0             	mov    %r14,%rax
    2467:	f3 48 ab             	rep stos %rax,%es:(%rdi)
    246a:	c5 f5 66 05 8e 0c 00 	vpcmpgtd 0xc8e(%rip),%ymm1,%ymm0        # 3100 <_IO_stdin_used+0x100>
    2471:	00 
    2472:	c4 e2 7d 17 c0       	vptest %ymm0,%ymm0
    2477:	0f 84 83 fe ff ff    	je     2300 <main+0x2d0>
    247d:	c4 c2 7d 8c 57 04    	vpmaskmovd 0x4(%r15),%ymm0,%ymm2
    2483:	c5 f5 fe ca          	vpaddd %ymm2,%ymm1,%ymm1
    2487:	c4 c2 7d 8e 4f 04    	vpmaskmovd %ymm1,%ymm0,0x4(%r15)
    248d:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
    2491:	c4 c2 7d 8e 0c 24    	vpmaskmovd %ymm1,%ymm0,(%r12)
    2497:	e9 64 fe ff ff       	jmpq   2300 <main+0x2d0>
    249c:	0f 1f 40 00          	nopl   0x0(%rax)
    24a0:	4c 89 e8             	mov    %r13,%rax
    24a3:	48 2b 44 24 38       	sub    0x38(%rsp),%rax
    24a8:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    24ad:	48 c1 f8 02          	sar    $0x2,%rax
    24b1:	0f 84 c9 00 00 00    	je     2580 <main+0x550>
    24b7:	48 c7 44 24 20 fc ff 	movq   $0xfffffffffffffffc,0x20(%rsp)
    24be:	ff ff 
    24c0:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    24c4:	48 39 d0             	cmp    %rdx,%rax
    24c7:	76 77                	jbe    2540 <main+0x510>
    24c9:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    24ce:	e8 ed eb ff ff       	callq  10c0 <_Znwm@plt>
    24d3:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    24d8:	49 89 c0             	mov    %rax,%r8
    24db:	48 01 c1             	add    %rax,%rcx
    24de:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    24e3:	48 8b 74 24 38       	mov    0x38(%rsp),%rsi
    24e8:	45 89 34 00          	mov    %r14d,(%r8,%rax,1)
    24ec:	4c 39 ee             	cmp    %r13,%rsi
    24ef:	74 7f                	je     2570 <main+0x540>
    24f1:	4c 89 c7             	mov    %r8,%rdi
    24f4:	48 89 c2             	mov    %rax,%rdx
    24f7:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    24fc:	49 89 c6             	mov    %rax,%r14
    24ff:	e8 4c ec ff ff       	callq  1150 <memmove@plt>
    2504:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    2509:	49 89 c0             	mov    %rax,%r8
    250c:	4f 8d 6c 30 04       	lea    0x4(%r8,%r14,1),%r13
    2511:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    2516:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
    251b:	48 89 4c 24 20       	mov    %rcx,0x20(%rsp)
    2520:	e8 7b eb ff ff       	callq  10a0 <_ZdlPv@plt>
    2525:	4c 8b 44 24 10       	mov    0x10(%rsp),%r8
    252a:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    252f:	4c 89 44 24 38       	mov    %r8,0x38(%rsp)
    2534:	e9 a7 fc ff ff       	jmpq   21e0 <main+0x1b0>
    2539:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2540:	48 be ff ff ff ff ff 	movabs $0x3fffffffffffffff,%rsi
    2547:	ff ff 3f 
    254a:	48 39 f2             	cmp    %rsi,%rdx
    254d:	0f 87 9a 00 00 00    	ja     25ed <main+0x5bd>
    2553:	48 85 d2             	test   %rdx,%rdx
    2556:	0f 85 9f 00 00 00    	jne    25fb <main+0x5cb>
    255c:	31 c9                	xor    %ecx,%ecx
    255e:	45 31 c0             	xor    %r8d,%r8d
    2561:	e9 78 ff ff ff       	jmpq   24de <main+0x4ae>
    2566:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    256d:	00 00 00 
    2570:	48 83 7c 24 38 00    	cmpq   $0x0,0x38(%rsp)
    2576:	4d 8d 6c 00 04       	lea    0x4(%r8,%rax,1),%r13
    257b:	74 b2                	je     252f <main+0x4ff>
    257d:	eb 92                	jmp    2511 <main+0x4e1>
    257f:	90                   	nop
    2580:	48 c7 44 24 20 04 00 	movq   $0x4,0x20(%rsp)
    2587:	00 00 
    2589:	e9 3b ff ff ff       	jmpq   24c9 <main+0x499>
    258e:	ba 07 00 00 00       	mov    $0x7,%edx
    2593:	48 8d 35 b0 0a 00 00 	lea    0xab0(%rip),%rsi        # 304a <_IO_stdin_used+0x4a>
    259a:	48 8d 3d 5f 2b 00 00 	lea    0x2b5f(%rip),%rdi        # 5100 <_ZSt4cout@@GLIBCXX_3.4>
    25a1:	e8 3a eb ff ff       	callq  10e0 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    25a6:	85 db                	test   %ebx,%ebx
    25a8:	48 8d 35 55 0a 00 00 	lea    0xa55(%rip),%rsi        # 3004 <_IO_stdin_used+0x4>
    25af:	48 8d 05 63 0a 00 00 	lea    0xa63(%rip),%rax        # 3019 <_IO_stdin_used+0x19>
    25b6:	48 0f 4e f0          	cmovle %rax,%rsi
    25ba:	48 8d 3d 3f 2b 00 00 	lea    0x2b3f(%rip),%rdi        # 5100 <_ZSt4cout@@GLIBCXX_3.4>
    25c1:	e8 ea ea ff ff       	callq  10b0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    25c6:	48 8d 3d 33 2b 00 00 	lea    0x2b33(%rip),%rdi        # 5100 <_ZSt4cout@@GLIBCXX_3.4>
    25cd:	e8 8e ea ff ff       	callq  1060 <_ZNSo5flushEv@plt>
    25d2:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    25d7:	e8 34 eb ff ff       	callq  1110 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    25dc:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
    25e0:	31 c0                	xor    %eax,%eax
    25e2:	5b                   	pop    %rbx
    25e3:	41 5c                	pop    %r12
    25e5:	41 5d                	pop    %r13
    25e7:	41 5e                	pop    %r14
    25e9:	41 5f                	pop    %r15
    25eb:	5d                   	pop    %rbp
    25ec:	c3                   	retq   
    25ed:	48 c7 44 24 20 fc ff 	movq   $0xfffffffffffffffc,0x20(%rsp)
    25f4:	ff ff 
    25f6:	e9 ce fe ff ff       	jmpq   24c9 <main+0x499>
    25fb:	48 c1 e0 03          	shl    $0x3,%rax
    25ff:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    2604:	e9 c0 fe ff ff       	jmpq   24c9 <main+0x499>
    2609:	48 89 c3             	mov    %rax,%rbx
    260c:	eb 05                	jmp    2613 <main+0x5e3>
    260e:	48 89 c3             	mov    %rax,%rbx
    2611:	eb 17                	jmp    262a <main+0x5fa>
    2613:	48 83 7c 24 38 00    	cmpq   $0x0,0x38(%rsp)
    2619:	74 37                	je     2652 <main+0x622>
    261b:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    2620:	c5 f8 77             	vzeroupper 
    2623:	e8 78 ea ff ff       	callq  10a0 <_ZdlPv@plt>
    2628:	eb 0d                	jmp    2637 <main+0x607>
    262a:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    262f:	c5 f8 77             	vzeroupper 
    2632:	e8 59 ea ff ff       	callq  1090 <_ZNSt13random_device7_M_finiEv@plt>
    2637:	48 89 df             	mov    %rbx,%rdi
    263a:	e8 41 eb ff ff       	callq  1180 <_Unwind_Resume@plt>
    263f:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
    2644:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    2649:	48 83 c0 10          	add    $0x10,%rax
    264d:	48 39 c7             	cmp    %rax,%rdi
    2650:	75 ce                	jne    2620 <main+0x5f0>
    2652:	c5 f8 77             	vzeroupper 
    2655:	eb e0                	jmp    2637 <main+0x607>
    2657:	48 89 c3             	mov    %rax,%rbx
    265a:	eb e3                	jmp    263f <main+0x60f>
    265c:	48 89 c3             	mov    %rax,%rbx
    265f:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    2664:	c5 f8 77             	vzeroupper 
    2667:	e8 a4 ea ff ff       	callq  1110 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    266c:	eb c9                	jmp    2637 <main+0x607>
    266e:	66 90                	xchg   %ax,%ax

0000000000002670 <set_fast_math>:
    2670:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
    2675:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
    267c:	00 
    267d:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
    2682:	c3                   	retq   
    2683:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    268a:	00 00 00 
    268d:	0f 1f 00             	nopl   (%rax)

0000000000002690 <_start>:
    2690:	31 ed                	xor    %ebp,%ebp
    2692:	49 89 d1             	mov    %rdx,%r9
    2695:	5e                   	pop    %rsi
    2696:	48 89 e2             	mov    %rsp,%rdx
    2699:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    269d:	50                   	push   %rax
    269e:	54                   	push   %rsp
    269f:	4c 8d 05 ba 06 00 00 	lea    0x6ba(%rip),%r8        # 2d60 <__libc_csu_fini>
    26a6:	48 8d 0d 43 06 00 00 	lea    0x643(%rip),%rcx        # 2cf0 <__libc_csu_init>
    26ad:	48 8d 3d 7c f9 ff ff 	lea    -0x684(%rip),%rdi        # 2030 <main>
    26b4:	ff 15 26 29 00 00    	callq  *0x2926(%rip)        # 4fe0 <__libc_start_main@GLIBC_2.2.5>
    26ba:	f4                   	hlt    
    26bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000026c0 <deregister_tm_clones>:
    26c0:	48 8d 3d 29 2a 00 00 	lea    0x2a29(%rip),%rdi        # 50f0 <__TMC_END__>
    26c7:	48 8d 05 22 2a 00 00 	lea    0x2a22(%rip),%rax        # 50f0 <__TMC_END__>
    26ce:	48 39 f8             	cmp    %rdi,%rax
    26d1:	74 15                	je     26e8 <deregister_tm_clones+0x28>
    26d3:	48 8b 05 fe 28 00 00 	mov    0x28fe(%rip),%rax        # 4fd8 <_ITM_deregisterTMCloneTable>
    26da:	48 85 c0             	test   %rax,%rax
    26dd:	74 09                	je     26e8 <deregister_tm_clones+0x28>
    26df:	ff e0                	jmpq   *%rax
    26e1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    26e8:	c3                   	retq   
    26e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000026f0 <register_tm_clones>:
    26f0:	48 8d 3d f9 29 00 00 	lea    0x29f9(%rip),%rdi        # 50f0 <__TMC_END__>
    26f7:	48 8d 35 f2 29 00 00 	lea    0x29f2(%rip),%rsi        # 50f0 <__TMC_END__>
    26fe:	48 29 fe             	sub    %rdi,%rsi
    2701:	48 c1 fe 03          	sar    $0x3,%rsi
    2705:	48 89 f0             	mov    %rsi,%rax
    2708:	48 c1 e8 3f          	shr    $0x3f,%rax
    270c:	48 01 c6             	add    %rax,%rsi
    270f:	48 d1 fe             	sar    %rsi
    2712:	74 14                	je     2728 <register_tm_clones+0x38>
    2714:	48 8b 05 d5 28 00 00 	mov    0x28d5(%rip),%rax        # 4ff0 <_ITM_registerTMCloneTable>
    271b:	48 85 c0             	test   %rax,%rax
    271e:	74 08                	je     2728 <register_tm_clones+0x38>
    2720:	ff e0                	jmpq   *%rax
    2722:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2728:	c3                   	retq   
    2729:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002730 <__do_global_dtors_aux>:
    2730:	80 3d 01 2c 00 00 00 	cmpb   $0x0,0x2c01(%rip)        # 5338 <completed.7389>
    2737:	75 2f                	jne    2768 <__do_global_dtors_aux+0x38>
    2739:	55                   	push   %rbp
    273a:	48 83 3d 8e 28 00 00 	cmpq   $0x0,0x288e(%rip)        # 4fd0 <__cxa_finalize@GLIBC_2.2.5>
    2741:	00 
    2742:	48 89 e5             	mov    %rsp,%rbp
    2745:	74 0c                	je     2753 <__do_global_dtors_aux+0x23>
    2747:	48 8b 3d 92 29 00 00 	mov    0x2992(%rip),%rdi        # 50e0 <__dso_handle>
    274e:	e8 5d ea ff ff       	callq  11b0 <__cxa_finalize@plt>
    2753:	e8 68 ff ff ff       	callq  26c0 <deregister_tm_clones>
    2758:	c6 05 d9 2b 00 00 01 	movb   $0x1,0x2bd9(%rip)        # 5338 <completed.7389>
    275f:	5d                   	pop    %rbp
    2760:	c3                   	retq   
    2761:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2768:	c3                   	retq   
    2769:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002770 <frame_dummy>:
    2770:	e9 7b ff ff ff       	jmpq   26f0 <register_tm_clones>
    2775:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    277c:	00 00 00 
    277f:	90                   	nop

0000000000002780 <_Z9descargarPiS_._omp_fn.1>:
    2780:	55                   	push   %rbp
    2781:	48 89 e5             	mov    %rsp,%rbp
    2784:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    2788:	4c 63 57 10          	movslq 0x10(%rdi),%r10
    278c:	4c 8b 47 08          	mov    0x8(%rdi),%r8
    2790:	4c 8b 0f             	mov    (%rdi),%r9
    2793:	c5 fd 6f 2d 25 5c 00 	vmovdqa 0x5c25(%rip),%ymm5        # 83c0 <_ZL6zeroes>
    279a:	00 
    279b:	44 89 d1             	mov    %r10d,%ecx
    279e:	41 8d 82 f7 3f 00 00 	lea    0x3ff7(%r10),%eax
    27a5:	41 8d ba 00 40 00 00 	lea    0x4000(%r10),%edi
    27ac:	c5 fd 6f 35 ec 5b 00 	vmovdqa 0x5bec(%rip),%ymm6        # 83a0 <_ZL4ones>
    27b3:	00 
    27b4:	83 e9 08             	sub    $0x8,%ecx
    27b7:	4a 8d 34 95 00 00 00 	lea    0x0(,%r10,4),%rsi
    27be:	00 
    27bf:	4c 89 d2             	mov    %r10,%rdx
    27c2:	c5 7d 6f 15 b6 3b 00 	vmovdqa 0x3bb6(%rip),%ymm10        # 6380 <_ZL8maskfff0>
    27c9:	00 
    27ca:	0f 49 c1             	cmovns %ecx,%eax
    27cd:	c5 7d 6f de          	vmovdqa %ymm6,%ymm11
    27d1:	b9 f8 ff 0f 00       	mov    $0xffff8,%ecx
    27d6:	c5 7d 6f 0d 82 3b 00 	vmovdqa 0x3b82(%rip),%ymm9        # 6360 <_ZL8mask000f>
    27dd:	00 
    27de:	c5 7d 6f c5          	vmovdqa %ymm5,%ymm8
    27e2:	c1 f8 0e             	sar    $0xe,%eax
    27e5:	81 ff f8 ff 0f 00    	cmp    $0xffff8,%edi
    27eb:	0f 4d f9             	cmovge %ecx,%edi
    27ee:	49 8d 0c 30          	lea    (%r8,%rsi,1),%rcx
    27f2:	c5 fd 6f 39          	vmovdqa (%rcx),%ymm7
    27f6:	c5 c5 66 c6          	vpcmpgtd %ymm6,%ymm7,%ymm0
    27fa:	c5 c5 66 cd          	vpcmpgtd %ymm5,%ymm7,%ymm1
    27fe:	c5 7d 6f e0          	vmovdqa %ymm0,%ymm12
    2802:	c5 fd db c1          	vpand  %ymm1,%ymm0,%ymm0
    2806:	c5 7d d7 d8          	vpmovmskb %ymm0,%r11d
    280a:	45 85 db             	test   %r11d,%r11d
    280d:	0f 84 ad 01 00 00    	je     29c0 <_Z9descargarPiS_._omp_fn.1+0x240>
    2813:	c5 7d 6f 2d 85 3b 00 	vmovdqa 0x3b85(%rip),%ymm13        # 63a0 <_ZL4MASK>
    281a:	00 
    281b:	c5 fd 6f cd          	vmovdqa %ymm5,%ymm1
    281f:	c5 fd 6f dd          	vmovdqa %ymm5,%ymm3
    2823:	c5 15 ef f6          	vpxor  %ymm6,%ymm13,%ymm14
    2827:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    282e:	00 00 
    2830:	c5 95 db d0          	vpand  %ymm0,%ymm13,%ymm2
    2834:	c5 8d db e0          	vpand  %ymm0,%ymm14,%ymm4
    2838:	c5 cd db c0          	vpand  %ymm0,%ymm6,%ymm0
    283c:	c5 ed fe c9          	vpaddd %ymm1,%ymm2,%ymm1
    2840:	c5 c5 fa d0          	vpsubd %ymm0,%ymm7,%ymm2
    2844:	c4 c1 6d 66 c0       	vpcmpgtd %ymm8,%ymm2,%ymm0
    2849:	c5 dd fe e3          	vpaddd %ymm3,%ymm4,%ymm4
    284d:	c5 fd 6f fa          	vmovdqa %ymm2,%ymm7
    2851:	c5 fd 6f dc          	vmovdqa %ymm4,%ymm3
    2855:	c5 9d db c0          	vpand  %ymm0,%ymm12,%ymm0
    2859:	c5 7d d7 d8          	vpmovmskb %ymm0,%r11d
    285d:	45 85 db             	test   %r11d,%r11d
    2860:	75 ce                	jne    2830 <_Z9descargarPiS_._omp_fn.1+0xb0>
    2862:	c4 e3 fd 00 c9 93    	vpermq $0x93,%ymm1,%ymm1
    2868:	c5 fd 7f 11          	vmovdqa %ymm2,(%rcx)
    286c:	c4 c1 75 db c2       	vpand  %ymm10,%ymm1,%ymm0
    2871:	c4 c1 75 db c9       	vpand  %ymm9,%ymm1,%ymm1
    2876:	c5 dd fe c0          	vpaddd %ymm0,%ymm4,%ymm0
    287a:	48 63 c8             	movslq %eax,%rcx
    287d:	4c 8d 1d dc 32 00 00 	lea    0x32dc(%rip),%r11        # 5b60 <left_border>
    2884:	48 89 c8             	mov    %rcx,%rax
    2887:	48 c1 e0 05          	shl    $0x5,%rax
    288b:	c4 c1 7d 7f 04 03    	vmovdqa %ymm0,(%r11,%rax,1)
    2891:	8d 42 08             	lea    0x8(%rdx),%eax
    2894:	39 c7                	cmp    %eax,%edi
    2896:	0f 8e 44 01 00 00    	jle    29e0 <_Z9descargarPiS_._omp_fn.1+0x260>
    289c:	29 d7                	sub    %edx,%edi
    289e:	c4 e3 fd 00 c5 93    	vpermq $0x93,%ymm5,%ymm0
    28a4:	49 8d 74 30 20       	lea    0x20(%r8,%rsi,1),%rsi
    28a9:	c5 7d 6f 25 ef 3a 00 	vmovdqa 0x3aef(%rip),%ymm12        # 63a0 <_ZL4MASK>
    28b0:	00 
    28b1:	8d 57 f7             	lea    -0x9(%rdi),%edx
    28b4:	c4 c1 7d db f9       	vpand  %ymm9,%ymm0,%ymm7
    28b9:	c4 41 7d db fa       	vpand  %ymm10,%ymm0,%ymm15
    28be:	c1 ea 03             	shr    $0x3,%edx
    28c1:	c5 1d ef f6          	vpxor  %ymm6,%ymm12,%ymm14
    28c5:	c5 fd 7f 7c 24 e0    	vmovdqa %ymm7,-0x20(%rsp)
    28cb:	49 8d 04 d2          	lea    (%r10,%rdx,8),%rax
    28cf:	31 d2                	xor    %edx,%edx
    28d1:	49 8d 7c 80 40       	lea    0x40(%r8,%rax,4),%rdi
    28d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    28dd:	00 00 00 
    28e0:	c5 fd 6f 26          	vmovdqa (%rsi),%ymm4
    28e4:	c4 c1 5d 66 c3       	vpcmpgtd %ymm11,%ymm4,%ymm0
    28e9:	c4 c1 5d 66 d0       	vpcmpgtd %ymm8,%ymm4,%ymm2
    28ee:	c5 fd 6f f8          	vmovdqa %ymm0,%ymm7
    28f2:	c5 fd db c2          	vpand  %ymm2,%ymm0,%ymm0
    28f6:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    28fa:	85 c0                	test   %eax,%eax
    28fc:	0f 84 ae 00 00 00    	je     29b0 <_Z9descargarPiS_._omp_fn.1+0x230>
    2902:	c5 7d 6f ed          	vmovdqa %ymm5,%ymm13
    2906:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    290d:	00 00 00 
    2910:	c5 8d db d8          	vpand  %ymm0,%ymm14,%ymm3
    2914:	c5 e5 fe d9          	vpaddd %ymm1,%ymm3,%ymm3
    2918:	c5 9d db c8          	vpand  %ymm0,%ymm12,%ymm1
    291c:	c5 cd db c0          	vpand  %ymm0,%ymm6,%ymm0
    2920:	c5 dd fa d0          	vpsubd %ymm0,%ymm4,%ymm2
    2924:	c4 41 75 fe ed       	vpaddd %ymm13,%ymm1,%ymm13
    2929:	c5 fd 6f cb          	vmovdqa %ymm3,%ymm1
    292d:	c4 c1 6d 66 c0       	vpcmpgtd %ymm8,%ymm2,%ymm0
    2932:	c5 fd 6f e2          	vmovdqa %ymm2,%ymm4
    2936:	c5 c5 db c0          	vpand  %ymm0,%ymm7,%ymm0
    293a:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    293e:	85 c0                	test   %eax,%eax
    2940:	75 ce                	jne    2910 <_Z9descargarPiS_._omp_fn.1+0x190>
    2942:	c4 c3 fd 00 cd 93    	vpermq $0x93,%ymm13,%ymm1
    2948:	c5 fd 7f 16          	vmovdqa %ymm2,(%rsi)
    294c:	c4 c1 75 db c2       	vpand  %ymm10,%ymm1,%ymm0
    2951:	c4 c1 75 db c9       	vpand  %ymm9,%ymm1,%ymm1
    2956:	c5 e5 fe c0          	vpaddd %ymm0,%ymm3,%ymm0
    295a:	c5 fd 6f d8          	vmovdqa %ymm0,%ymm3
    295e:	c4 e2 7d 17 db       	vptest %ymm3,%ymm3
    2963:	74 1c                	je     2981 <_Z9descargarPiS_._omp_fn.1+0x201>
    2965:	c5 fd fe 46 fc       	vpaddd -0x4(%rsi),%ymm0,%ymm0
    296a:	c5 fe 7f 46 fc       	vmovdqu %ymm0,-0x4(%rsi)
    296f:	c4 c1 7d 66 c3       	vpcmpgtd %ymm11,%ymm0,%ymm0
    2974:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    2978:	f3 0f b8 c0          	popcnt %eax,%eax
    297c:	c1 f8 02             	sar    $0x2,%eax
    297f:	01 c2                	add    %eax,%edx
    2981:	48 83 c6 20          	add    $0x20,%rsi
    2985:	48 39 f7             	cmp    %rsi,%rdi
    2988:	0f 85 52 ff ff ff    	jne    28e0 <_Z9descargarPiS_._omp_fn.1+0x160>
    298e:	48 89 c8             	mov    %rcx,%rax
    2991:	48 8d 35 c8 29 00 00 	lea    0x29c8(%rip),%rsi        # 5360 <right_border>
    2998:	41 89 14 89          	mov    %edx,(%r9,%rcx,4)
    299c:	48 c1 e0 05          	shl    $0x5,%rax
    29a0:	c5 fd 7f 0c 06       	vmovdqa %ymm1,(%rsi,%rax,1)
    29a5:	c5 f8 77             	vzeroupper 
    29a8:	c9                   	leaveq 
    29a9:	c3                   	retq   
    29aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    29b0:	c5 85 fe c1          	vpaddd %ymm1,%ymm15,%ymm0
    29b4:	c5 fd 6f 4c 24 e0    	vmovdqa -0x20(%rsp),%ymm1
    29ba:	c5 fd 6f d8          	vmovdqa %ymm0,%ymm3
    29be:	eb 9e                	jmp    295e <_Z9descargarPiS_._omp_fn.1+0x1de>
    29c0:	c4 e3 fd 00 cd 93    	vpermq $0x93,%ymm5,%ymm1
    29c6:	c4 c1 75 db c2       	vpand  %ymm10,%ymm1,%ymm0
    29cb:	c4 c1 75 db c9       	vpand  %ymm9,%ymm1,%ymm1
    29d0:	c5 fd fe c5          	vpaddd %ymm5,%ymm0,%ymm0
    29d4:	e9 a1 fe ff ff       	jmpq   287a <_Z9descargarPiS_._omp_fn.1+0xfa>
    29d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    29e0:	31 d2                	xor    %edx,%edx
    29e2:	eb aa                	jmp    298e <_Z9descargarPiS_._omp_fn.1+0x20e>
    29e4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    29eb:	00 00 00 00 
    29ef:	90                   	nop

00000000000029f0 <_Z9descargarPiS_._omp_fn.0>:
    29f0:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    29f5:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    29f9:	41 ff 72 f8          	pushq  -0x8(%r10)
    29fd:	55                   	push   %rbp
    29fe:	48 89 e5             	mov    %rsp,%rbp
    2a01:	41 57                	push   %r15
    2a03:	41 56                	push   %r14
    2a05:	41 55                	push   %r13
    2a07:	41 54                	push   %r12
    2a09:	49 89 fc             	mov    %rdi,%r12
    2a0c:	41 52                	push   %r10
    2a0e:	53                   	push   %rbx
    2a0f:	48 83 ec 40          	sub    $0x40,%rsp
    2a13:	48 8b 47 08          	mov    0x8(%rdi),%rax
    2a17:	48 8b 1f             	mov    (%rdi),%rbx
    2a1a:	48 89 45 a8          	mov    %rax,-0x58(%rbp)
    2a1e:	e8 0d e6 ff ff       	callq  1030 <GOMP_single_start@plt>
    2a23:	84 c0                	test   %al,%al
    2a25:	74 6b                	je     2a92 <_Z9descargarPiS_._omp_fn.0+0xa2>
    2a27:	41 bf 08 00 00 00    	mov    $0x8,%r15d
    2a2d:	4c 8d 75 b0          	lea    -0x50(%rbp),%r14
    2a31:	4c 8d 2d 48 fd ff ff 	lea    -0x2b8(%rip),%r13        # 2780 <_Z9descargarPiS_._omp_fn.1>
    2a38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2a3f:	00 
    2a40:	48 83 ec 08          	sub    $0x8,%rsp
    2a44:	49 8b 54 24 18       	mov    0x18(%r12),%rdx
    2a49:	44 89 7d c0          	mov    %r15d,-0x40(%rbp)
    2a4d:	4c 89 f6             	mov    %r14,%rsi
    2a50:	6a 00                	pushq  $0x0
    2a52:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    2a58:	b9 18 00 00 00       	mov    $0x18,%ecx
    2a5d:	4c 89 ef             	mov    %r13,%rdi
    2a60:	6a 00                	pushq  $0x0
    2a62:	41 b8 08 00 00 00    	mov    $0x8,%r8d
    2a68:	41 81 c7 00 40 00 00 	add    $0x4000,%r15d
    2a6f:	6a 00                	pushq  $0x0
    2a71:	48 89 55 b0          	mov    %rdx,-0x50(%rbp)
    2a75:	31 d2                	xor    %edx,%edx
    2a77:	48 89 5d b8          	mov    %rbx,-0x48(%rbp)
    2a7b:	e8 d0 e5 ff ff       	callq  1050 <GOMP_task@plt>
    2a80:	48 83 c4 20          	add    $0x20,%rsp
    2a84:	41 81 ff 08 00 10 00 	cmp    $0x100008,%r15d
    2a8b:	75 b3                	jne    2a40 <_Z9descargarPiS_._omp_fn.0+0x50>
    2a8d:	e8 5e e6 ff ff       	callq  10f0 <GOMP_taskwait@plt>
    2a92:	e8 f9 e6 ff ff       	callq  1190 <GOMP_barrier@plt>
    2a97:	e8 94 e5 ff ff       	callq  1030 <GOMP_single_start@plt>
    2a9c:	84 c0                	test   %al,%al
    2a9e:	0f 84 8c 00 00 00    	je     2b30 <_Z9descargarPiS_._omp_fn.0+0x140>
    2aa4:	4d 8b 4c 24 10       	mov    0x10(%r12),%r9
    2aa9:	48 8d 93 e0 ff 3f 00 	lea    0x3fffe0(%rbx),%rdx
    2ab0:	be f9 ff 0f 00       	mov    $0xffff9,%esi
    2ab5:	eb 36                	jmp    2aed <_Z9descargarPiS_._omp_fn.0+0xfd>
    2ab7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    2abe:	00 00 
    2ac0:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    2ac4:	8b 84 b0 78 00 c0 ff 	mov    -0x3fff88(%rax,%rsi,4),%eax
    2acb:	03 42 fc             	add    -0x4(%rdx),%eax
    2ace:	83 f8 01             	cmp    $0x1,%eax
    2ad1:	89 42 fc             	mov    %eax,-0x4(%rdx)
    2ad4:	0f 9f c0             	setg   %al
    2ad7:	48 ff c6             	inc    %rsi
    2ada:	48 83 c2 04          	add    $0x4,%rdx
    2ade:	0f b6 c0             	movzbl %al,%eax
    2ae1:	41 01 01             	add    %eax,(%r9)
    2ae4:	48 81 fe 01 00 10 00 	cmp    $0x100001,%rsi
    2aeb:	74 43                	je     2b30 <_Z9descargarPiS_._omp_fn.0+0x140>
    2aed:	83 3a 01             	cmpl   $0x1,(%rdx)
    2af0:	7e ce                	jle    2ac0 <_Z9descargarPiS_._omp_fn.0+0xd0>
    2af2:	48 89 f0             	mov    %rsi,%rax
    2af5:	48 8b 7d a8          	mov    -0x58(%rbp),%rdi
    2af9:	83 e0 1f             	and    $0x1f,%eax
    2afc:	48 8d 3c 87          	lea    (%rdi,%rax,4),%rdi
    2b00:	31 c0                	xor    %eax,%eax
    2b02:	8b 0f                	mov    (%rdi),%ecx
    2b04:	44 8d 41 01          	lea    0x1(%rcx),%r8d
    2b08:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2b0f:	00 
    2b10:	41 8d 0c 00          	lea    (%r8,%rax,1),%ecx
    2b14:	ff c0                	inc    %eax
    2b16:	89 0f                	mov    %ecx,(%rdi)
    2b18:	39 02                	cmp    %eax,(%rdx)
    2b1a:	7f f4                	jg     2b10 <_Z9descargarPiS_._omp_fn.0+0x120>
    2b1c:	c7 02 00 00 00 00    	movl   $0x0,(%rdx)
    2b22:	eb 9c                	jmp    2ac0 <_Z9descargarPiS_._omp_fn.0+0xd0>
    2b24:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    2b2b:	00 00 00 00 
    2b2f:	90                   	nop
    2b30:	e8 5b e6 ff ff       	callq  1190 <GOMP_barrier@plt>
    2b35:	e8 f6 e4 ff ff       	callq  1030 <GOMP_single_start@plt>
    2b3a:	84 c0                	test   %al,%al
    2b3c:	0f 85 6e 01 00 00    	jne    2cb0 <_Z9descargarPiS_._omp_fn.0+0x2c0>
    2b42:	e8 59 e6 ff ff       	callq  11a0 <omp_get_num_threads@plt>
    2b47:	41 89 c5             	mov    %eax,%r13d
    2b4a:	e8 f1 e5 ff ff       	callq  1140 <omp_get_thread_num@plt>
    2b4f:	89 c1                	mov    %eax,%ecx
    2b51:	b8 3f 00 00 00       	mov    $0x3f,%eax
    2b56:	99                   	cltd   
    2b57:	41 f7 fd             	idiv   %r13d
    2b5a:	39 d1                	cmp    %edx,%ecx
    2b5c:	0f 8c 83 01 00 00    	jl     2ce5 <_Z9descargarPiS_._omp_fn.0+0x2f5>
    2b62:	0f af c8             	imul   %eax,%ecx
    2b65:	31 ff                	xor    %edi,%edi
    2b67:	01 d1                	add    %edx,%ecx
    2b69:	8d 14 08             	lea    (%rax,%rcx,1),%edx
    2b6c:	39 d1                	cmp    %edx,%ecx
    2b6e:	0f 8d 8f 00 00 00    	jge    2c03 <_Z9descargarPiS_._omp_fn.0+0x213>
    2b74:	4c 63 c1             	movslq %ecx,%r8
    2b77:	8d 51 01             	lea    0x1(%rcx),%edx
    2b7a:	ff c8                	dec    %eax
    2b7c:	49 8b 4c 24 18       	mov    0x18(%r12),%rcx
    2b81:	4c 01 c0             	add    %r8,%rax
    2b84:	c1 e2 0e             	shl    $0xe,%edx
    2b87:	c5 fd 6f 0d 11 58 00 	vmovdqa 0x5811(%rip),%ymm1        # 83a0 <_ZL4ones>
    2b8e:	00 
    2b8f:	4c 8d 0d ea 2f 00 00 	lea    0x2fea(%rip),%r9        # 5b80 <left_border+0x20>
    2b96:	4a 8d 34 81          	lea    (%rcx,%r8,4),%rsi
    2b9a:	48 63 d2             	movslq %edx,%rdx
    2b9d:	4c 89 c1             	mov    %r8,%rcx
    2ba0:	48 c1 e0 10          	shl    $0x10,%rax
    2ba4:	48 8d 54 93 1c       	lea    0x1c(%rbx,%rdx,4),%rdx
    2ba9:	48 c1 e1 05          	shl    $0x5,%rcx
    2bad:	4c 8d 94 03 1c 00 02 	lea    0x2001c(%rbx,%rax,1),%r10
    2bb4:	00 
    2bb5:	4c 8d 05 a4 27 00 00 	lea    0x27a4(%rip),%r8        # 5360 <right_border>
    2bbc:	0f 1f 40 00          	nopl   0x0(%rax)
    2bc0:	c4 c1 7d 6f 14 09    	vmovdqa (%r9,%rcx,1),%ymm2
    2bc6:	c4 c1 6d fe 04 08    	vpaddd (%r8,%rcx,1),%ymm2,%ymm0
    2bcc:	48 83 c6 04          	add    $0x4,%rsi
    2bd0:	48 83 c1 20          	add    $0x20,%rcx
    2bd4:	c5 fd fe 02          	vpaddd (%rdx),%ymm0,%ymm0
    2bd8:	48 81 c2 00 00 01 00 	add    $0x10000,%rdx
    2bdf:	c5 fe 7f 82 00 00 ff 	vmovdqu %ymm0,-0x10000(%rdx)
    2be6:	ff 
    2be7:	c5 fd 66 c1          	vpcmpgtd %ymm1,%ymm0,%ymm0
    2beb:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    2bef:	f3 0f b8 c0          	popcnt %eax,%eax
    2bf3:	c1 f8 02             	sar    $0x2,%eax
    2bf6:	03 46 fc             	add    -0x4(%rsi),%eax
    2bf9:	01 c7                	add    %eax,%edi
    2bfb:	49 39 d2             	cmp    %rdx,%r10
    2bfe:	75 c0                	jne    2bc0 <_Z9descargarPiS_._omp_fn.0+0x1d0>
    2c00:	c5 f8 77             	vzeroupper 
    2c03:	49 8b 44 24 10       	mov    0x10(%r12),%rax
    2c08:	f0 01 38             	lock add %edi,(%rax)
    2c0b:	e8 80 e5 ff ff       	callq  1190 <GOMP_barrier@plt>
    2c10:	e8 1b e4 ff ff       	callq  1030 <GOMP_single_start@plt>
    2c15:	84 c0                	test   %al,%al
    2c17:	74 79                	je     2c92 <_Z9descargarPiS_._omp_fn.0+0x2a2>
    2c19:	49 8b 44 24 10       	mov    0x10(%r12),%rax
    2c1e:	31 d2                	xor    %edx,%edx
    2c20:	83 bb e0 ff 3f 00 01 	cmpl   $0x1,0x3fffe0(%rbx)
    2c27:	0f 9f c2             	setg   %dl
    2c2a:	31 c9                	xor    %ecx,%ecx
    2c2c:	c5 f9 6f 05 0c 2f 00 	vmovdqa 0x2f0c(%rip),%xmm0        # 5b40 <right_border+0x7e0>
    2c33:	00 
    2c34:	8b 30                	mov    (%rax),%esi
    2c36:	29 d6                	sub    %edx,%esi
    2c38:	89 30                	mov    %esi,(%rax)
    2c3a:	83 bb dc ff 3f 00 01 	cmpl   $0x1,0x3fffdc(%rbx)
    2c41:	89 f2                	mov    %esi,%edx
    2c43:	0f 9f c1             	setg   %cl
    2c46:	29 ca                	sub    %ecx,%edx
    2c48:	c4 e3 79 16 c1 01    	vpextrd $0x1,%xmm0,%ecx
    2c4e:	89 10                	mov    %edx,(%rax)
    2c50:	c5 f9 7e c2          	vmovd  %xmm0,%edx
    2c54:	03 93 e0 ff 3f 00    	add    0x3fffe0(%rbx),%edx
    2c5a:	01 8b dc ff 3f 00    	add    %ecx,0x3fffdc(%rbx)
    2c60:	83 fa 01             	cmp    $0x1,%edx
    2c63:	89 93 e0 ff 3f 00    	mov    %edx,0x3fffe0(%rbx)
    2c69:	0f 9f c2             	setg   %dl
    2c6c:	8b 08                	mov    (%rax),%ecx
    2c6e:	0f b6 d2             	movzbl %dl,%edx
    2c71:	01 d1                	add    %edx,%ecx
    2c73:	31 d2                	xor    %edx,%edx
    2c75:	89 08                	mov    %ecx,(%rax)
    2c77:	83 bb dc ff 3f 00 01 	cmpl   $0x1,0x3fffdc(%rbx)
    2c7e:	0f 9f c2             	setg   %dl
    2c81:	01 ca                	add    %ecx,%edx
    2c83:	49 8b 4c 24 18       	mov    0x18(%r12),%rcx
    2c88:	89 10                	mov    %edx,(%rax)
    2c8a:	03 91 fc 00 00 00    	add    0xfc(%rcx),%edx
    2c90:	89 10                	mov    %edx,(%rax)
    2c92:	e8 f9 e4 ff ff       	callq  1190 <GOMP_barrier@plt>
    2c97:	48 8d 65 d0          	lea    -0x30(%rbp),%rsp
    2c9b:	5b                   	pop    %rbx
    2c9c:	41 5a                	pop    %r10
    2c9e:	41 5c                	pop    %r12
    2ca0:	41 5d                	pop    %r13
    2ca2:	41 5e                	pop    %r14
    2ca4:	41 5f                	pop    %r15
    2ca6:	5d                   	pop    %rbp
    2ca7:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    2cab:	c3                   	retq   
    2cac:	0f 1f 40 00          	nopl   0x0(%rax)
    2cb0:	c5 fd 6f 1d a8 2e 00 	vmovdqa 0x2ea8(%rip),%ymm3        # 5b60 <left_border>
    2cb7:	00 
    2cb8:	c5 e5 fe 43 1c       	vpaddd 0x1c(%rbx),%ymm3,%ymm0
    2cbd:	49 8b 54 24 10       	mov    0x10(%r12),%rdx
    2cc2:	c5 fe 7f 43 1c       	vmovdqu %ymm0,0x1c(%rbx)
    2cc7:	c5 fd 66 05 d1 56 00 	vpcmpgtd 0x56d1(%rip),%ymm0,%ymm0        # 83a0 <_ZL4ones>
    2cce:	00 
    2ccf:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
    2cd3:	f3 0f b8 c0          	popcnt %eax,%eax
    2cd7:	c1 f8 02             	sar    $0x2,%eax
    2cda:	f0 01 02             	lock add %eax,(%rdx)
    2cdd:	c5 f8 77             	vzeroupper 
    2ce0:	e9 5d fe ff ff       	jmpq   2b42 <_Z9descargarPiS_._omp_fn.0+0x152>
    2ce5:	ff c0                	inc    %eax
    2ce7:	31 d2                	xor    %edx,%edx
    2ce9:	e9 74 fe ff ff       	jmpq   2b62 <_Z9descargarPiS_._omp_fn.0+0x172>
    2cee:	66 90                	xchg   %ax,%ax

0000000000002cf0 <__libc_csu_init>:
    2cf0:	41 57                	push   %r15
    2cf2:	41 56                	push   %r14
    2cf4:	49 89 d7             	mov    %rdx,%r15
    2cf7:	41 55                	push   %r13
    2cf9:	41 54                	push   %r12
    2cfb:	4c 8d 25 7e 20 00 00 	lea    0x207e(%rip),%r12        # 4d80 <__frame_dummy_init_array_entry>
    2d02:	55                   	push   %rbp
    2d03:	48 8d 2d 8e 20 00 00 	lea    0x208e(%rip),%rbp        # 4d98 <__init_array_end>
    2d0a:	53                   	push   %rbx
    2d0b:	41 89 fd             	mov    %edi,%r13d
    2d0e:	49 89 f6             	mov    %rsi,%r14
    2d11:	4c 29 e5             	sub    %r12,%rbp
    2d14:	48 83 ec 08          	sub    $0x8,%rsp
    2d18:	48 c1 fd 03          	sar    $0x3,%rbp
    2d1c:	e8 df e2 ff ff       	callq  1000 <_init>
    2d21:	48 85 ed             	test   %rbp,%rbp
    2d24:	74 20                	je     2d46 <__libc_csu_init+0x56>
    2d26:	31 db                	xor    %ebx,%ebx
    2d28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    2d2f:	00 
    2d30:	4c 89 fa             	mov    %r15,%rdx
    2d33:	4c 89 f6             	mov    %r14,%rsi
    2d36:	44 89 ef             	mov    %r13d,%edi
    2d39:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
    2d3d:	48 83 c3 01          	add    $0x1,%rbx
    2d41:	48 39 dd             	cmp    %rbx,%rbp
    2d44:	75 ea                	jne    2d30 <__libc_csu_init+0x40>
    2d46:	48 83 c4 08          	add    $0x8,%rsp
    2d4a:	5b                   	pop    %rbx
    2d4b:	5d                   	pop    %rbp
    2d4c:	41 5c                	pop    %r12
    2d4e:	41 5d                	pop    %r13
    2d50:	41 5e                	pop    %r14
    2d52:	41 5f                	pop    %r15
    2d54:	c3                   	retq   
    2d55:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    2d5c:	00 00 00 00 

0000000000002d60 <__libc_csu_fini>:
    2d60:	f3 c3                	repz retq 

Disassembly of section .fini:

0000000000002d64 <_fini>:
    2d64:	48 83 ec 08          	sub    $0x8,%rsp
    2d68:	48 83 c4 08          	add    $0x8,%rsp
    2d6c:	c3                   	retq   
