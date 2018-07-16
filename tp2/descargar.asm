    # empieza descargar()
    1190:	c7 03 56 34 12 00    	movl   $0x123456,(%rbx)
    1196:	31 f6                	xor    %esi,%esi
    1198:	ba 00 00 40 00       	mov    $0x400000,%edx
    119d:	4c 89 ef             	mov    %r13,%rdi
    11a0:	e8 3b fb ff ff       	callq  ce0 <memset@plt>
    11a5:	31 f6                	xor    %esi,%esi
    11a7:	eb 17                	jmp    11c0 <main+0x380>
    11a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    11b0:	48 ff c6             	inc    %rsi
    11b3:	48 81 fe 00 00 10 00 	cmp    $0x100000,%rsi
    11ba:	0f 84 99 00 00 00    	je     1259 <main+0x419>
    
    # empieza el bucle principal
    11c0:	44 8b 04 b3          	mov    (%rbx,%rsi,4),%r8d
    11c4:	41 89 f2             	mov    %esi,%r10d
    11c7:	41 83 f8 01          	cmp    $0x1,%r8d
    11cb:	7e e3                	jle    11b0 <main+0x370>
    11cd:	48 8b 0d 34 11 20 00 	mov    0x201134(%rip),%rcx        # 202308 <_ZL9generator>
    11d4:	4c 8d 0d 2d 11 20 00 	lea    0x20112d(%rip),%r9        # 202308 <_ZL9generator>
    11db:	31 ff                	xor    %edi,%edi
    11dd:	0f 1f 00             	nopl   (%rax)
    11e0:	4c 69 d9 a7 41 00 00 	imul   $0x41a7,%rcx,%r11
    11e7:	4c 89 d8             	mov    %r11,%rax
    11ea:	4c 89 d9             	mov    %r11,%rcx
    11ed:	49 f7 e6             	mul    %r14
    11f0:	48 29 d1             	sub    %rdx,%rcx
    11f3:	48 d1 e9             	shr    %rcx
    11f6:	48 01 ca             	add    %rcx,%rdx
    11f9:	48 89 d1             	mov    %rdx,%rcx
    11fc:	48 c1 e9 1e          	shr    $0x1e,%rcx
    1200:	48 89 c8             	mov    %rcx,%rax
    1203:	48 c1 e0 1f          	shl    $0x1f,%rax
    1207:	48 29 c8             	sub    %rcx,%rax
    120a:	4c 89 d9             	mov    %r11,%rcx
    120d:	48 29 c1             	sub    %rax,%rcx
    1210:	48 8d 41 ff          	lea    -0x1(%rcx),%rax
    1214:	48 3d fb ff ff 7f    	cmp    $0x7ffffffb,%rax
    121a:	77 c4                	ja     11e0 <main+0x3a0>
    121c:	48 d1 e8             	shr    %rax
    121f:	ff c7                	inc    %edi
    1221:	49 89 09             	mov    %rcx,(%r9)
    1224:	49 f7 e7             	mul    %r15
    1227:	48 c1 ea 1c          	shr    $0x1c,%rdx
    122b:	41 8d 84 52 ff ff 0f 	lea    0xfffff(%r10,%rdx,2),%eax
    1232:	00 
    1233:	25 ff ff 0f 00       	and    $0xfffff,%eax
    1238:	41 ff 44 85 00       	incl   0x0(%r13,%rax,4)
    123d:	41 39 f8             	cmp    %edi,%r8d
    1240:	75 9e                	jne    11e0 <main+0x3a0>
    1242:	c7 04 b3 00 00 00 00 	movl   $0x0,(%rbx,%rsi,4)
    1249:	48 ff c6             	inc    %rsi
    124c:	48 81 fe 00 00 10 00 	cmp    $0x100000,%rsi
    1253:	0f 85 67 ff ff ff    	jne    11c0 <main+0x380>
    
    # empieza el bucle secundario
    1259:	c7 03 77 77 00 00    	movl   $0x7777,(%rbx)
    125f:	31 c0                	xor    %eax,%eax
    1261:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
    1265:	90                   	nop
    1266:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    126d:	00 00 00 
    1270:	c4 c1 7d 6f 74 05 00 	vmovdqa 0x0(%r13,%rax,1),%ymm6
    1277:	c5 cd fe 04 03       	vpaddd (%rbx,%rax,1),%ymm6,%ymm0
    127c:	c5 fd 7f 04 03       	vmovdqa %ymm0,(%rbx,%rax,1)
    1281:	48 83 c0 20          	add    $0x20,%rax
    1285:	c5 fd 66 05 d3 06 00 	vpcmpgtd 0x6d3(%rip),%ymm0,%ymm0        # 1960 <_IO_stdin_used+0x140>
    128c:	00 
    128d:	c5 f5 fa c8          	vpsubd %ymm0,%ymm1,%ymm1
    1291:	48 3d 00 00 40 00    	cmp    $0x400000,%rax
    1297:	75 d7                	jne    1270 <main+0x430>
    1299:	c4 e3 7d 39 c8 01    	vextracti128 $0x1,%ymm1,%xmm0
    129f:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    12a4:	c7 03 f0 fe 00 00    	movl   $0xfef0,(%rbx)
    # termina la funcion descargar()
