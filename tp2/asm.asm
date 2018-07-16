
tiny_manna:     file format elf64-x86-64


Disassembly of section .interp:

0000000000000238 <.interp>:
 238:	2f                   	(bad)  
 239:	6c                   	insb   (%dx),%es:(%rdi)
 23a:	69 62 36 34 2f 6c 64 	imul   $0x646c2f34,0x36(%rdx),%esp
 241:	2d 6c 69 6e 75       	sub    $0x756e696c,%eax
 246:	78 2d                	js     275 <_init-0xaa3>
 248:	78 38                	js     282 <_init-0xa96>
 24a:	36 2d 36 34 2e 73    	ss sub $0x732e3436,%eax
 250:	6f                   	outsl  %ds:(%rsi),(%dx)
 251:	2e 32 00             	xor    %cs:(%rax),%al

Disassembly of section .note.ABI-tag:

0000000000000254 <.note.ABI-tag>:
 254:	04 00                	add    $0x0,%al
 256:	00 00                	add    %al,(%rax)
 258:	10 00                	adc    %al,(%rax)
 25a:	00 00                	add    %al,(%rax)
 25c:	01 00                	add    %eax,(%rax)
 25e:	00 00                	add    %al,(%rax)
 260:	47                   	rex.RXB
 261:	4e 55                	rex.WRX push %rbp
 263:	00 00                	add    %al,(%rax)
 265:	00 00                	add    %al,(%rax)
 267:	00 03                	add    %al,(%rbx)
 269:	00 00                	add    %al,(%rax)
 26b:	00 02                	add    %al,(%rdx)
 26d:	00 00                	add    %al,(%rax)
 26f:	00 00                	add    %al,(%rax)
 271:	00 00                	add    %al,(%rax)
	...

Disassembly of section .note.gnu.build-id:

0000000000000274 <.note.gnu.build-id>:
 274:	04 00                	add    $0x0,%al
 276:	00 00                	add    %al,(%rax)
 278:	14 00                	adc    $0x0,%al
 27a:	00 00                	add    %al,(%rax)
 27c:	03 00                	add    (%rax),%eax
 27e:	00 00                	add    %al,(%rax)
 280:	47                   	rex.RXB
 281:	4e 55                	rex.WRX push %rbp
 283:	00 0b                	add    %cl,(%rbx)
 285:	ef                   	out    %eax,(%dx)
 286:	2d 0e 65 77 23       	sub    $0x2377650e,%eax
 28b:	60                   	(bad)  
 28c:	ce                   	(bad)  
 28d:	b6 95                	mov    $0x95,%dh
 28f:	e5 8c                	in     $0x8c,%eax
 291:	e7 ff                	out    %eax,$0xff
 293:	ef                   	out    %eax,(%dx)
 294:	91                   	xchg   %eax,%ecx
 295:	64 50                	fs push %rax
 297:	24                   	.byte 0x24

Disassembly of section .gnu.hash:

0000000000000298 <.gnu.hash>:
 298:	02 00                	add    (%rax),%al
 29a:	00 00                	add    %al,(%rax)
 29c:	1c 00                	sbb    $0x0,%al
 29e:	00 00                	add    %al,(%rax)
 2a0:	01 00                	add    %eax,(%rax)
 2a2:	00 00                	add    %al,(%rax)
 2a4:	06                   	(bad)  
 2a5:	00 00                	add    %al,(%rax)
 2a7:	00 00                	add    %al,(%rax)
 2a9:	00 10                	add    %dl,(%rax)
 2ab:	02 01                	add    (%rcx),%al
 2ad:	00 04 00             	add    %al,(%rax,%rax,1)
 2b0:	1c 00                	sbb    $0x0,%al
 2b2:	00 00                	add    %al,(%rax)
 2b4:	00 00                	add    %al,(%rax)
 2b6:	00 00                	add    %al,(%rax)
 2b8:	14 98                	adc    $0x98,%al
 2ba:	0c 43                	or     $0x43,%al
 2bc:	73 96                	jae    254 <_init-0xac4>
 2be:	07                   	(bad)  
 2bf:	02                   	.byte 0x2

Disassembly of section .dynsym:

00000000000002c0 <.dynsym>:
	...
 2d8:	d5                   	(bad)  
 2d9:	00 00                	add    %al,(%rax)
 2db:	00 12                	add    %dl,(%rdx)
	...
 2ed:	00 00                	add    %al,(%rax)
 2ef:	00 77 00             	add    %dh,0x0(%rdi)
 2f2:	00 00                	add    %al,(%rax)
 2f4:	22 00                	and    (%rax),%al
	...
 306:	00 00                	add    %al,(%rax)
 308:	46 02 00             	rex.RX add (%rax),%r8b
 30b:	00 12                	add    %dl,(%rdx)
	...
 31d:	00 00                	add    %al,(%rax)
 31f:	00 f6                	add    %dh,%dh
 321:	01 00                	add    %eax,(%rax)
 323:	00 12                	add    %dl,(%rdx)
	...
 335:	00 00                	add    %al,(%rax)
 337:	00 a6 01 00 00 12    	add    %ah,0x12000001(%rsi)
	...
 34d:	00 00                	add    %al,(%rax)
 34f:	00 c8                	add    %cl,%al
 351:	00 00                	add    %al,(%rax)
 353:	00 12                	add    %dl,(%rdx)
	...
 365:	00 00                	add    %al,(%rax)
 367:	00 7c 01 00          	add    %bh,0x0(%rcx,%rax,1)
 36b:	00 12                	add    %dl,(%rdx)
	...
 37d:	00 00                	add    %al,(%rax)
 37f:	00 87 01 00 00 12    	add    %al,0x12000001(%rdi)
	...
 395:	00 00                	add    %al,(%rax)
 397:	00 54 01 00          	add    %dl,0x0(%rcx,%rax,1)
 39b:	00 12                	add    %dl,(%rdx)
	...
 3ad:	00 00                	add    %al,(%rax)
 3af:	00 81 01 00 00 12    	add    %al,0x12000001(%rcx)
	...
 3c5:	00 00                	add    %al,(%rax)
 3c7:	00 be 01 00 00 12    	add    %bh,0x12000001(%rsi)
	...
 3dd:	00 00                	add    %al,(%rax)
 3df:	00 d4                	add    %dl,%ah
 3e1:	02 00                	add    (%rax),%al
 3e3:	00 12                	add    %dl,(%rdx)
	...
 3f5:	00 00                	add    %al,(%rax)
 3f7:	00 04 02             	add    %al,(%rdx,%rax,1)
 3fa:	00 00                	add    %al,(%rax)
 3fc:	12 00                	adc    (%rax),%al
	...
 40e:	00 00                	add    %al,(%rax)
 410:	57                   	push   %rdi
 411:	02 00                	add    (%rax),%al
 413:	00 12                	add    %dl,(%rdx)
	...
 425:	00 00                	add    %al,(%rax)
 427:	00 5b 01             	add    %bl,0x1(%rbx)
 42a:	00 00                	add    %al,(%rax)
 42c:	12 00                	adc    (%rax),%al
	...
 43e:	00 00                	add    %al,(%rax)
 440:	a5                   	movsl  %ds:(%rsi),%es:(%rdi)
 441:	02 00                	add    (%rax),%al
 443:	00 12                	add    %dl,(%rdx)
	...
 455:	00 00                	add    %al,(%rax)
 457:	00 98 00 00 00 12    	add    %bl,0x12000000(%rax)
	...
 46d:	00 00                	add    %al,(%rax)
 46f:	00 da                	add    %bl,%dl
 471:	02 00                	add    (%rax),%al
 473:	00 12                	add    %dl,(%rdx)
	...
 485:	00 00                	add    %al,(%rax)
 487:	00 f1                	add    %dh,%cl
 489:	02 00                	add    (%rax),%al
 48b:	00 12                	add    %dl,(%rdx)
	...
 4a1:	01 00                	add    %eax,(%rax)
 4a3:	00 12                	add    %dl,(%rdx)
	...
 4b5:	00 00                	add    %al,(%rax)
 4b7:	00 4d 02             	add    %cl,0x2(%rbp)
 4ba:	00 00                	add    %al,(%rax)
 4bc:	12 00                	adc    (%rax),%al
	...
 4ce:	00 00                	add    %al,(%rax)
 4d0:	1f                   	(bad)  
 4d1:	00 00                	add    %al,(%rax)
 4d3:	00 20                	add    %ah,(%rax)
	...
 4e5:	00 00                	add    %al,(%rax)
 4e7:	00 e2                	add    %ah,%dl
 4e9:	02 00                	add    (%rax),%al
 4eb:	00 12                	add    %dl,(%rdx)
	...
 4fd:	00 00                	add    %al,(%rax)
 4ff:	00 86 00 00 00 12    	add    %al,0x12000000(%rsi)
	...
 515:	00 00                	add    %al,(%rax)
 517:	00 10                	add    %dl,(%rax)
 519:	00 00                	add    %al,(%rax)
 51b:	00 20                	add    %ah,(%rax)
	...
 52d:	00 00                	add    %al,(%rax)
 52f:	00 3b                	add    %bh,(%rbx)
 531:	00 00                	add    %al,(%rax)
 533:	00 20                	add    %ah,(%rax)
	...
 545:	00 00                	add    %al,(%rax)
 547:	00 b0 00 00 00 12    	add    %dh,0x12000000(%rax)
	...
 55d:	00 00                	add    %al,(%rax)
 55f:	00 b4 01 00 00 11 00 	add    %dh,0x110000(%rcx,%rax,1)
 566:	1a 00                	sbb    (%rax),%al
 568:	e0 20                	loopne 58a <_init-0x78e>
 56a:	20 00                	and    %al,(%rax)
 56c:	00 00                	add    %al,(%rax)
 56e:	00 00                	add    %al,(%rax)
 570:	10 01                	adc    %al,(%rcx)
 572:	00 00                	add    %al,(%rax)
 574:	00 00                	add    %al,(%rax)
 576:	00 00                	add    %al,(%rax)
 578:	f7 00 00 00 11 00    	testl  $0x110000,(%rax)
 57e:	1a 00                	sbb    (%rax),%al
 580:	00 22                	add    %ah,(%rdx)
 582:	20 00                	and    %al,(%rax)
 584:	00 00                	add    %al,(%rax)
 586:	00 00                	add    %al,(%rax)
 588:	18 01                	sbb    %al,(%rcx)
 58a:	00 00                	add    %al,(%rax)
 58c:	00 00                	add    %al,(%rax)
	...

Disassembly of section .dynstr:

0000000000000590 <.dynstr>:
 590:	00 6c 69 62          	add    %ch,0x62(%rcx,%rbp,2)
 594:	73 74                	jae    60a <_init-0x70e>
 596:	64 63 2b             	movslq %fs:(%rbx),%ebp
 599:	2b 2e                	sub    (%rsi),%ebp
 59b:	73 6f                	jae    60c <_init-0x70c>
 59d:	2e 36 00 5f 5f       	cs add %bl,%ss:0x5f(%rdi)
 5a2:	67 6d                	insl   (%dx),%es:(%edi)
 5a4:	6f                   	outsl  %ds:(%rsi),(%dx)
 5a5:	6e                   	outsb  %ds:(%rsi),(%dx)
 5a6:	5f                   	pop    %rdi
 5a7:	73 74                	jae    61d <_init-0x6fb>
 5a9:	61                   	(bad)  
 5aa:	72 74                	jb     620 <_init-0x6f8>
 5ac:	5f                   	pop    %rdi
 5ad:	5f                   	pop    %rdi
 5ae:	00 5f 49             	add    %bl,0x49(%rdi)
 5b1:	54                   	push   %rsp
 5b2:	4d 5f                	rex.WRB pop %r15
 5b4:	64 65 72 65          	fs gs jb 61d <_init-0x6fb>
 5b8:	67 69 73 74 65 72 54 	imul   $0x4d547265,0x74(%ebx),%esi
 5bf:	4d 
 5c0:	43 6c                	rex.XB insb (%dx),%es:(%rdi)
 5c2:	6f                   	outsl  %ds:(%rsi),(%dx)
 5c3:	6e                   	outsb  %ds:(%rsi),(%dx)
 5c4:	65 54                	gs push %rsp
 5c6:	61                   	(bad)  
 5c7:	62                   	(bad)  
 5c8:	6c                   	insb   (%dx),%es:(%rdi)
 5c9:	65 00 5f 49          	add    %bl,%gs:0x49(%rdi)
 5cd:	54                   	push   %rsp
 5ce:	4d 5f                	rex.WRB pop %r15
 5d0:	72 65                	jb     637 <_init-0x6e1>
 5d2:	67 69 73 74 65 72 54 	imul   $0x4d547265,0x74(%ebx),%esi
 5d9:	4d 
 5da:	43 6c                	rex.XB insb (%dx),%es:(%rdi)
 5dc:	6f                   	outsl  %ds:(%rsi),(%dx)
 5dd:	6e                   	outsb  %ds:(%rsi),(%dx)
 5de:	65 54                	gs push %rsp
 5e0:	61                   	(bad)  
 5e1:	62                   	(bad)  
 5e2:	6c                   	insb   (%dx),%es:(%rdi)
 5e3:	65 00 6c 69 62       	add    %ch,%gs:0x62(%rcx,%rbp,2)
 5e8:	6d                   	insl   (%dx),%es:(%rdi)
 5e9:	2e 73 6f             	jae,pn 65b <_init-0x6bd>
 5ec:	2e 36 00 6c 69 62    	cs add %ch,%ss:0x62(%rcx,%rbp,2)
 5f2:	67 63 63 5f          	movslq 0x5f(%ebx),%esp
 5f6:	73 2e                	jae    626 <_init-0x6f2>
 5f8:	73 6f                	jae    669 <_init-0x6af>
 5fa:	2e 31 00             	xor    %eax,%cs:(%rax)
 5fd:	6c                   	insb   (%dx),%es:(%rdi)
 5fe:	69 62 63 2e 73 6f 2e 	imul   $0x2e6f732e,0x63(%rdx),%esp
 605:	36 00 5f 5f          	add    %bl,%ss:0x5f(%rdi)
 609:	63 78 61             	movslq 0x61(%rax),%edi
 60c:	5f                   	pop    %rdi
 60d:	66 69 6e 61 6c 69    	imul   $0x696c,0x61(%rsi),%bp
 613:	7a 65                	jp     67a <_init-0x69e>
 615:	00 5f 5f             	add    %bl,0x5f(%rdi)
 618:	6c                   	insb   (%dx),%es:(%rdi)
 619:	69 62 63 5f 73 74 61 	imul   $0x6174735f,0x63(%rdx),%esp
 620:	72 74                	jb     696 <_init-0x682>
 622:	5f                   	pop    %rdi
 623:	6d                   	insl   (%dx),%es:(%rdi)
 624:	61                   	(bad)  
 625:	69 6e 00 5f 5a 4e 53 	imul   $0x534e5a5f,0x0(%rsi),%ebp
 62c:	74 38                	je     666 <_init-0x6b2>
 62e:	69 6f 73 5f 62 61 73 	imul   $0x7361625f,0x73(%rdi),%ebp
 635:	65 34 49             	gs xor $0x49,%al
 638:	6e                   	outsb  %ds:(%rsi),(%dx)
 639:	69 74 43 31 45 76 00 	imul   $0x5f007645,0x31(%rbx,%rax,2),%esi
 640:	5f 
 641:	5a                   	pop    %rdx
 642:	4e 53                	rex.WRX push %rbx
 644:	74 38                	je     67e <_init-0x69a>
 646:	69 6f 73 5f 62 61 73 	imul   $0x7361625f,0x73(%rdi),%ebp
 64d:	65 34 49             	gs xor $0x49,%al
 650:	6e                   	outsb  %ds:(%rsi),(%dx)
 651:	69 74 44 31 45 76 00 	imul   $0x5f007645,0x31(%rsp,%rax,2),%esi
 658:	5f 
 659:	5f                   	pop    %rdi
 65a:	63 78 61             	movslq 0x61(%rax),%edi
 65d:	5f                   	pop    %rdi
 65e:	61                   	(bad)  
 65f:	74 65                	je     6c6 <_init-0x652>
 661:	78 69                	js     6cc <_init-0x64c>
 663:	74 00                	je     665 <_init-0x6b3>
 665:	5f                   	pop    %rdi
 666:	5a                   	pop    %rdx
 667:	4e 53                	rex.WRX push %rbx
 669:	74 38                	je     6a3 <_init-0x675>
 66b:	69 6f 73 5f 62 61 73 	imul   $0x7361625f,0x73(%rdi),%ebp
 672:	65 31 35 73 79 6e 63 	xor    %esi,%gs:0x636e7973(%rip)        # 636e7fec <_end+0x634e5cbc>
 679:	5f                   	pop    %rdi
 67a:	77 69                	ja     6e5 <_init-0x633>
 67c:	74 68                	je     6e6 <_init-0x632>
 67e:	5f                   	pop    %rdi
 67f:	73 74                	jae    6f5 <_init-0x623>
 681:	64 69 6f 45 62 00 5f 	imul   $0x5a5f0062,%fs:0x45(%rdi),%ebp
 688:	5a 
 689:	53                   	push   %rbx
 68a:	74 33                	je     6bf <_init-0x659>
 68c:	63 69 6e             	movslq 0x6e(%rcx),%ebp
 68f:	00 5f 5a             	add    %bl,0x5a(%rdi)
 692:	4e 53                	rex.WRX push %rbx
 694:	74 31                	je     6c7 <_init-0x651>
 696:	33 72 61             	xor    0x61(%rdx),%esi
 699:	6e                   	outsb  %ds:(%rsi),(%dx)
 69a:	64 6f                	outsl  %fs:(%rsi),(%dx)
 69c:	6d                   	insl   (%dx),%es:(%rdi)
 69d:	5f                   	pop    %rdi
 69e:	64 65 76 69          	fs gs jbe 70b <_init-0x60d>
 6a2:	63 65 37             	movslq 0x37(%rbp),%esp
 6a5:	5f                   	pop    %rdi
 6a6:	4d 5f                	rex.WRB pop %r15
 6a8:	69 6e 69 74 45 52 4b 	imul   $0x4b524574,0x69(%rsi),%ebp
 6af:	4e 53                	rex.WRX push %rbx
 6b1:	74 37                	je     6ea <_init-0x62e>
 6b3:	5f                   	pop    %rdi
 6b4:	5f                   	pop    %rdi
 6b5:	63 78 78             	movslq 0x78(%rax),%edi
 6b8:	31 31                	xor    %esi,(%rcx)
 6ba:	31 32                	xor    %esi,(%rdx)
 6bc:	62 61                	(bad)  
 6be:	73 69                	jae    729 <_init-0x5ef>
 6c0:	63 5f 73             	movslq 0x73(%rdi),%ebx
 6c3:	74 72                	je     737 <_init-0x5e1>
 6c5:	69 6e 67 49 63 53 74 	imul   $0x74536349,0x67(%rsi),%ebp
 6cc:	31 31                	xor    %esi,(%rcx)
 6ce:	63 68 61             	movslq 0x61(%rax),%ebp
 6d1:	72 5f                	jb     732 <_init-0x5e6>
 6d3:	74 72                	je     747 <_init-0x5d1>
 6d5:	61                   	(bad)  
 6d6:	69 74 73 49 63 45 53 	imul   $0x61534563,0x49(%rbx,%rsi,2),%esi
 6dd:	61 
 6de:	49 63 45 45          	movslq 0x45(%r13),%rax
 6e2:	45 00 5f 5a          	add    %r11b,0x5a(%r15)
 6e6:	64 6c                	fs insb (%dx),%es:(%rdi)
 6e8:	50                   	push   %rax
 6e9:	76 00                	jbe    6eb <_init-0x62d>
 6eb:	5f                   	pop    %rdi
 6ec:	5a                   	pop    %rdx
 6ed:	4e 53                	rex.WRX push %rbx
 6ef:	74 31                	je     722 <_init-0x5f6>
 6f1:	33 72 61             	xor    0x61(%rdx),%esi
 6f4:	6e                   	outsb  %ds:(%rsi),(%dx)
 6f5:	64 6f                	outsl  %fs:(%rsi),(%dx)
 6f7:	6d                   	insl   (%dx),%es:(%rdi)
 6f8:	5f                   	pop    %rdi
 6f9:	64 65 76 69          	fs gs jbe 766 <_init-0x5b2>
 6fd:	63 65 39             	movslq 0x39(%rbp),%esp
 700:	5f                   	pop    %rdi
 701:	4d 5f                	rex.WRB pop %r15
 703:	67 65 74 76          	addr32 gs je 77d <_init-0x59b>
 707:	61                   	(bad)  
 708:	6c                   	insb   (%dx),%es:(%rdi)
 709:	45 76 00             	rex.RB jbe 70c <_init-0x60c>
 70c:	74 69                	je     777 <_init-0x5a1>
 70e:	6d                   	insl   (%dx),%es:(%rdi)
 70f:	65 00 73 72          	add    %dh,%gs:0x72(%rbx)
 713:	61                   	(bad)  
 714:	6e                   	outsb  %ds:(%rsi),(%dx)
 715:	64 00 5f 5a          	add    %bl,%fs:0x5a(%rdi)
 719:	4e 53                	rex.WRX push %rbx
 71b:	74 31                	je     74e <_init-0x5ca>
 71d:	33 72 61             	xor    0x61(%rdx),%esi
 720:	6e                   	outsb  %ds:(%rsi),(%dx)
 721:	64 6f                	outsl  %fs:(%rsi),(%dx)
 723:	6d                   	insl   (%dx),%es:(%rdi)
 724:	5f                   	pop    %rdi
 725:	64 65 76 69          	fs gs jbe 792 <_init-0x586>
 729:	63 65 37             	movslq 0x37(%rbp),%esp
 72c:	5f                   	pop    %rdi
 72d:	4d 5f                	rex.WRB pop %r15
 72f:	66 69 6e 69 45 76    	imul   $0x7645,0x69(%rsi),%bp
 735:	00 61 6c             	add    %ah,0x6c(%rcx)
 738:	69 67 6e 65 64 5f 61 	imul   $0x615f6465,0x6e(%rdi),%esp
 73f:	6c                   	insb   (%dx),%es:(%rdi)
 740:	6c                   	insb   (%dx),%es:(%rdi)
 741:	6f                   	outsl  %ds:(%rsi),(%dx)
 742:	63 00                	movslq (%rax),%eax
 744:	5f                   	pop    %rdi
 745:	5a                   	pop    %rdx
 746:	53                   	push   %rbx
 747:	74 34                	je     77d <_init-0x59b>
 749:	63 6f 75             	movslq 0x75(%rdi),%ebp
 74c:	74 00                	je     74e <_init-0x5ca>
 74e:	5f                   	pop    %rdi
 74f:	5a                   	pop    %rdx
 750:	53                   	push   %rbx
 751:	74 6c                	je     7bf <_init-0x559>
 753:	73 49                	jae    79e <_init-0x57a>
 755:	53                   	push   %rbx
 756:	74 31                	je     789 <_init-0x58f>
 758:	31 63 68             	xor    %esp,0x68(%rbx)
 75b:	61                   	(bad)  
 75c:	72 5f                	jb     7bd <_init-0x55b>
 75e:	74 72                	je     7d2 <_init-0x546>
 760:	61                   	(bad)  
 761:	69 74 73 49 63 45 45 	imul   $0x52454563,0x49(%rbx,%rsi,2),%esi
 768:	52 
 769:	53                   	push   %rbx
 76a:	74 31                	je     79d <_init-0x57b>
 76c:	33 62 61             	xor    0x61(%rdx),%esp
 76f:	73 69                	jae    7da <_init-0x53e>
 771:	63 5f 6f             	movslq 0x6f(%rdi),%ebx
 774:	73 74                	jae    7ea <_init-0x52e>
 776:	72 65                	jb     7dd <_init-0x53b>
 778:	61                   	(bad)  
 779:	6d                   	insl   (%dx),%es:(%rdi)
 77a:	49 63 54 5f 45       	movslq 0x45(%r15,%rbx,2),%rdx
 77f:	53                   	push   %rbx
 780:	35 5f 50 4b 63       	xor    $0x634b505f,%eax
 785:	00 5f 5a             	add    %bl,0x5a(%rdi)
 788:	4e 53                	rex.WRX push %rbx
 78a:	6f                   	outsl  %ds:(%rsi),(%dx)
 78b:	35 66 6c 75 73       	xor    $0x73756c66,%eax
 790:	68 45 76 00 5f       	pushq  $0x5f007645
 795:	5a                   	pop    %rdx
 796:	4e 53                	rex.WRX push %rbx
 798:	74 31                	je     7cb <_init-0x54d>
 79a:	34 62                	xor    $0x62,%al
 79c:	61                   	(bad)  
 79d:	73 69                	jae    808 <_init-0x510>
 79f:	63 5f 6f             	movslq 0x6f(%rdi),%ebx
 7a2:	66 73 74             	data16 jae 819 <_init-0x4ff>
 7a5:	72 65                	jb     80c <_init-0x50c>
 7a7:	61                   	(bad)  
 7a8:	6d                   	insl   (%dx),%es:(%rdi)
 7a9:	49 63 53 74          	movslq 0x74(%r11),%rdx
 7ad:	31 31                	xor    %esi,(%rcx)
 7af:	63 68 61             	movslq 0x61(%rax),%ebp
 7b2:	72 5f                	jb     813 <_init-0x505>
 7b4:	74 72                	je     828 <_init-0x4f0>
 7b6:	61                   	(bad)  
 7b7:	69 74 73 49 63 45 45 	imul   $0x43454563,0x49(%rbx,%rsi,2),%esi
 7be:	43 
 7bf:	31 45 50             	xor    %eax,0x50(%rbp)
 7c2:	4b 63 53 74          	rex.WXB movslq 0x74(%r11),%rdx
 7c6:	31 33                	xor    %esi,(%rbx)
 7c8:	5f                   	pop    %rdi
 7c9:	49 6f                	rex.WB outsl %ds:(%rsi),(%dx)
 7cb:	73 5f                	jae    82c <_init-0x4ec>
 7cd:	4f 70 65             	rex.WRXB jo 835 <_init-0x4e3>
 7d0:	6e                   	outsb  %ds:(%rsi),(%dx)
 7d1:	6d                   	insl   (%dx),%es:(%rdi)
 7d2:	6f                   	outsl  %ds:(%rsi),(%dx)
 7d3:	64 65 00 6d 65       	fs add %ch,%gs:0x65(%rbp)
 7d8:	6d                   	insl   (%dx),%es:(%rdi)
 7d9:	73 65                	jae    840 <_init-0x4d8>
 7db:	74 00                	je     7dd <_init-0x53b>
 7dd:	5f                   	pop    %rdi
 7de:	5a                   	pop    %rdx
 7df:	4e 53                	rex.WRX push %rbx
 7e1:	6f                   	outsl  %ds:(%rsi),(%dx)
 7e2:	6c                   	insb   (%dx),%es:(%rdi)
 7e3:	73 45                	jae    82a <_init-0x4ee>
 7e5:	69 00 5f 5a 53 74    	imul   $0x74535a5f,(%rax),%eax
 7eb:	31 36                	xor    %esi,(%rsi)
 7ed:	5f                   	pop    %rdi
 7ee:	5f                   	pop    %rdi
 7ef:	6f                   	outsl  %ds:(%rsi),(%dx)
 7f0:	73 74                	jae    866 <_init-0x4b2>
 7f2:	72 65                	jb     859 <_init-0x4bf>
 7f4:	61                   	(bad)  
 7f5:	6d                   	insl   (%dx),%es:(%rdi)
 7f6:	5f                   	pop    %rdi
 7f7:	69 6e 73 65 72 74 49 	imul   $0x49747265,0x73(%rsi),%ebp
 7fe:	63 53 74             	movslq 0x74(%rbx),%edx
 801:	31 31                	xor    %esi,(%rcx)
 803:	63 68 61             	movslq 0x61(%rax),%ebp
 806:	72 5f                	jb     867 <_init-0x4b1>
 808:	74 72                	je     87c <_init-0x49c>
 80a:	61                   	(bad)  
 80b:	69 74 73 49 63 45 45 	imul   $0x52454563,0x49(%rbx,%rsi,2),%esi
 812:	52 
 813:	53                   	push   %rbx
 814:	74 31                	je     847 <_init-0x4d1>
 816:	33 62 61             	xor    0x61(%rdx),%esp
 819:	73 69                	jae    884 <_init-0x494>
 81b:	63 5f 6f             	movslq 0x6f(%rdi),%ebx
 81e:	73 74                	jae    894 <_init-0x484>
 820:	72 65                	jb     887 <_init-0x491>
 822:	61                   	(bad)  
 823:	6d                   	insl   (%dx),%es:(%rdi)
 824:	49 54                	rex.WB push %r12
 826:	5f                   	pop    %rdi
 827:	54                   	push   %rsp
 828:	30 5f 45             	xor    %bl,0x45(%rdi)
 82b:	53                   	push   %rbx
 82c:	36 5f                	ss pop %rdi
 82e:	50                   	push   %rax
 82f:	4b 53                	rex.WXB push %r11
 831:	33 5f 6c             	xor    0x6c(%rdi),%ebx
 834:	00 5f 5a             	add    %bl,0x5a(%rdi)
 837:	4e 53                	rex.WRX push %rbx
 839:	74 31                	je     86c <_init-0x4ac>
 83b:	34 62                	xor    $0x62,%al
 83d:	61                   	(bad)  
 83e:	73 69                	jae    8a9 <_init-0x46f>
 840:	63 5f 6f             	movslq 0x6f(%rdi),%ebx
 843:	66 73 74             	data16 jae 8ba <_init-0x45e>
 846:	72 65                	jb     8ad <_init-0x46b>
 848:	61                   	(bad)  
 849:	6d                   	insl   (%dx),%es:(%rdi)
 84a:	49 63 53 74          	movslq 0x74(%r11),%rdx
 84e:	31 31                	xor    %esi,(%rcx)
 850:	63 68 61             	movslq 0x61(%rax),%ebp
 853:	72 5f                	jb     8b4 <_init-0x464>
 855:	74 72                	je     8c9 <_init-0x44f>
 857:	61                   	(bad)  
 858:	69 74 73 49 63 45 45 	imul   $0x44454563,0x49(%rbx,%rsi,2),%esi
 85f:	44 
 860:	31 45 76             	xor    %eax,0x76(%rbp)
 863:	00 5f 5a             	add    %bl,0x5a(%rdi)
 866:	6e                   	outsb  %ds:(%rsi),(%dx)
 867:	77 6d                	ja     8d6 <_init-0x442>
 869:	00 6d 65             	add    %ch,0x65(%rbp)
 86c:	6d                   	insl   (%dx),%es:(%rdi)
 86d:	6d                   	insl   (%dx),%es:(%rdi)
 86e:	6f                   	outsl  %ds:(%rsi),(%dx)
 86f:	76 65                	jbe    8d6 <_init-0x442>
 871:	00 5f 55             	add    %bl,0x55(%rdi)
 874:	6e                   	outsb  %ds:(%rsi),(%dx)
 875:	77 69                	ja     8e0 <_init-0x438>
 877:	6e                   	outsb  %ds:(%rsi),(%dx)
 878:	64 5f                	fs pop %rdi
 87a:	52                   	push   %rdx
 87b:	65 73 75             	gs jae 8f3 <_init-0x425>
 87e:	6d                   	insl   (%dx),%es:(%rdi)
 87f:	65 00 5f 5f          	add    %bl,%gs:0x5f(%rdi)
 883:	67 78 78             	addr32 js 8fe <_init-0x41a>
 886:	5f                   	pop    %rdi
 887:	70 65                	jo     8ee <_init-0x42a>
 889:	72 73                	jb     8fe <_init-0x41a>
 88b:	6f                   	outsl  %ds:(%rsi),(%dx)
 88c:	6e                   	outsb  %ds:(%rsi),(%dx)
 88d:	61                   	(bad)  
 88e:	6c                   	insb   (%dx),%es:(%rdi)
 88f:	69 74 79 5f 76 30 00 	imul   $0x47003076,0x5f(%rcx,%rdi,2),%esi
 896:	47 
 897:	43                   	rex.XB
 898:	43 5f                	rex.XB pop %r15
 89a:	33 2e                	xor    (%rsi),%ebp
 89c:	30 00                	xor    %al,(%rax)
 89e:	47                   	rex.RXB
 89f:	4c                   	rex.WR
 8a0:	49                   	rex.WB
 8a1:	42                   	rex.X
 8a2:	43 5f                	rex.XB pop %r15
 8a4:	32 2e                	xor    (%rsi),%ch
 8a6:	31 36                	xor    %esi,(%rsi)
 8a8:	00 47 4c             	add    %al,0x4c(%rdi)
 8ab:	49                   	rex.WB
 8ac:	42                   	rex.X
 8ad:	43 5f                	rex.XB pop %r15
 8af:	32 2e                	xor    (%rsi),%ch
 8b1:	32 2e                	xor    (%rsi),%ch
 8b3:	35 00 47 4c 49       	xor    $0x494c4700,%eax
 8b8:	42                   	rex.X
 8b9:	43 58                	rex.XB pop %r8
 8bb:	58                   	pop    %rax
 8bc:	5f                   	pop    %rdi
 8bd:	33 2e                	xor    (%rsi),%ebp
 8bf:	34 2e                	xor    $0x2e,%al
 8c1:	32 31                	xor    (%rcx),%dh
 8c3:	00 43 58             	add    %al,0x58(%rbx)
 8c6:	58                   	pop    %rax
 8c7:	41                   	rex.B
 8c8:	42                   	rex.X
 8c9:	49 5f                	rex.WB pop %r15
 8cb:	31 2e                	xor    %ebp,(%rsi)
 8cd:	33 00                	xor    (%rax),%eax
 8cf:	47                   	rex.RXB
 8d0:	4c                   	rex.WR
 8d1:	49                   	rex.WB
 8d2:	42                   	rex.X
 8d3:	43 58                	rex.XB pop %r8
 8d5:	58                   	pop    %rax
 8d6:	5f                   	pop    %rdi
 8d7:	33 2e                	xor    (%rsi),%ebp
 8d9:	34 2e                	xor    $0x2e,%al
 8db:	39 00                	cmp    %eax,(%rax)
 8dd:	47                   	rex.RXB
 8de:	4c                   	rex.WR
 8df:	49                   	rex.WB
 8e0:	42                   	rex.X
 8e1:	43 58                	rex.XB pop %r8
 8e3:	58                   	pop    %rax
 8e4:	5f                   	pop    %rdi
 8e5:	33 2e                	xor    (%rsi),%ebp
 8e7:	34 2e                	xor    $0x2e,%al
 8e9:	31 38                	xor    %edi,(%rax)
 8eb:	00 47 4c             	add    %al,0x4c(%rdi)
 8ee:	49                   	rex.WB
 8ef:	42                   	rex.X
 8f0:	43 58                	rex.XB pop %r8
 8f2:	58                   	pop    %rax
 8f3:	5f                   	pop    %rdi
 8f4:	33 2e                	xor    (%rsi),%ebp
 8f6:	34 00                	xor    $0x0,%al

Disassembly of section .gnu.version:

00000000000008f8 <.gnu.version>:
 8f8:	00 00                	add    %al,(%rax)
 8fa:	02 00                	add    (%rax),%al
 8fc:	03 00                	add    (%rax),%eax
 8fe:	03 00                	add    (%rax),%eax
 900:	02 00                	add    (%rax),%al
 902:	04 00                	add    $0x0,%al
 904:	03 00                	add    (%rax),%eax
 906:	03 00                	add    (%rax),%eax
 908:	05 00 02 00 03       	add    $0x3000200,%eax
 90d:	00 02                	add    %al,(%rdx)
 90f:	00 02                	add    %al,(%rdx)
 911:	00 02                	add    %al,(%rdx)
 913:	00 06                	add    %al,(%rsi)
 915:	00 05 00 02 00 02    	add    %al,0x2000200(%rip)        # 2000b1b <_end+0x1dfe7eb>
 91b:	00 03                	add    %al,(%rbx)
 91d:	00 07                	add    %al,(%rdi)
 91f:	00 08                	add    %cl,(%rax)
 921:	00 02                	add    %al,(%rdx)
 923:	00 00                	add    %al,(%rax)
 925:	00 09                	add    %cl,(%rcx)
 927:	00 03                	add    %al,(%rbx)
 929:	00 00                	add    %al,(%rax)
 92b:	00 00                	add    %al,(%rax)
 92d:	00 02                	add    %al,(%rdx)
 92f:	00 02                	add    %al,(%rdx)
 931:	00 02                	add    %al,(%rdx)
	...

Disassembly of section .gnu.version_r:

0000000000000938 <.gnu.version_r>:
 938:	01 00                	add    %eax,(%rax)
 93a:	01 00                	add    %eax,(%rax)
 93c:	5f                   	pop    %rdi
 93d:	00 00                	add    %al,(%rax)
 93f:	00 10                	add    %dl,(%rax)
 941:	00 00                	add    %al,(%rax)
 943:	00 20                	add    %ah,(%rax)
 945:	00 00                	add    %al,(%rax)
 947:	00 50 26             	add    %dl,0x26(%rax)
 94a:	79 0b                	jns    957 <_init-0x3c1>
 94c:	00 00                	add    %al,(%rax)
 94e:	09 00                	or     %eax,(%rax)
 950:	06                   	(bad)  
 951:	03 00                	add    (%rax),%eax
 953:	00 00                	add    %al,(%rax)
 955:	00 00                	add    %al,(%rax)
 957:	00 01                	add    %al,(%rcx)
 959:	00 02                	add    %al,(%rdx)
 95b:	00 6d 00             	add    %ch,0x0(%rbp)
 95e:	00 00                	add    %al,(%rax)
 960:	10 00                	adc    %al,(%rax)
 962:	00 00                	add    %al,(%rax)
 964:	30 00                	xor    %al,(%rax)
 966:	00 00                	add    %al,(%rax)
 968:	96                   	xchg   %eax,%esi
 969:	91                   	xchg   %eax,%ecx
 96a:	96                   	xchg   %eax,%esi
 96b:	06                   	(bad)  
 96c:	00 00                	add    %al,(%rax)
 96e:	04 00                	add    $0x0,%al
 970:	0e                   	(bad)  
 971:	03 00                	add    (%rax),%eax
 973:	00 10                	add    %dl,(%rax)
 975:	00 00                	add    %al,(%rax)
 977:	00 75 1a             	add    %dh,0x1a(%rbp)
 97a:	69 09 00 00 03 00    	imul   $0x30000,(%rcx),%ecx
 980:	19 03                	sbb    %eax,(%rbx)
 982:	00 00                	add    %al,(%rax)
 984:	00 00                	add    %al,(%rax)
 986:	00 00                	add    %al,(%rax)
 988:	01 00                	add    %eax,(%rax)
 98a:	05 00 01 00 00       	add    $0x100,%eax
 98f:	00 10                	add    %dl,(%rax)
 991:	00 00                	add    %al,(%rax)
 993:	00 00                	add    %al,(%rax)
 995:	00 00                	add    %al,(%rax)
 997:	00 71 f8             	add    %dh,-0x8(%rcx)
 99a:	97                   	xchg   %eax,%edi
 99b:	02 00                	add    (%rax),%al
 99d:	00 08                	add    %cl,(%rax)
 99f:	00 25 03 00 00 10    	add    %ah,0x10000003(%rip)        # 100009a8 <_end+0xfdfe678>
 9a5:	00 00                	add    %al,(%rax)
 9a7:	00 d3                	add    %dl,%bl
 9a9:	af                   	scas   %es:(%rdi),%eax
 9aa:	6b 05 00 00 07 00 34 	imul   $0x34,0x70000(%rip),%eax        # 709b1 <__FRAME_END__+0x6ed45>
 9b1:	03 00                	add    (%rax),%eax
 9b3:	00 10                	add    %dl,(%rax)
 9b5:	00 00                	add    %al,(%rax)
 9b7:	00 89 7f 29 02 00    	add    %cl,0x2297f(%rcx)
 9bd:	00 06                	add    %al,(%rsi)
 9bf:	00 3f                	add    %bh,(%rdi)
 9c1:	03 00                	add    (%rax),%eax
 9c3:	00 10                	add    %dl,(%rax)
 9c5:	00 00                	add    %al,(%rax)
 9c7:	00 68 f8             	add    %ch,-0x8(%rax)
 9ca:	97                   	xchg   %eax,%edi
 9cb:	02 00                	add    (%rax),%al
 9cd:	00 05 00 4d 03 00    	add    %al,0x34d00(%rip)        # 356d3 <__FRAME_END__+0x33a67>
 9d3:	00 10                	add    %dl,(%rax)
 9d5:	00 00                	add    %al,(%rax)
 9d7:	00 74 29 92          	add    %dh,-0x6e(%rcx,%rbp,1)
 9db:	08 00                	or     %al,(%rax)
 9dd:	00 02                	add    %al,(%rdx)
 9df:	00 5c 03 00          	add    %bl,0x0(%rbx,%rax,1)
 9e3:	00 00                	add    %al,(%rax)
 9e5:	00 00                	add    %al,(%rax)
	...

Disassembly of section .rela.dyn:

00000000000009e8 <.rela.dyn>:
 9e8:	a0 1d 20 00 00 00 00 	movabs 0x80000000000201d,%al
 9ef:	00 08 
 9f1:	00 00                	add    %al,(%rax)
 9f3:	00 00                	add    %al,(%rax)
 9f5:	00 00                	add    %al,(%rax)
 9f7:	00 90 16 00 00 00    	add    %dl,0x16(%rax)
 9fd:	00 00                	add    %al,(%rax)
 9ff:	00 a8 1d 20 00 00    	add    %ch,0x201d(%rax)
 a05:	00 00                	add    %al,(%rax)
 a07:	00 08                	add    %cl,(%rax)
 a09:	00 00                	add    %al,(%rax)
 a0b:	00 00                	add    %al,(%rax)
 a0d:	00 00                	add    %al,(%rax)
 a0f:	00 90 0e 00 00 00    	add    %dl,0xe(%rax)
 a15:	00 00                	add    %al,(%rax)
 a17:	00 b0 1d 20 00 00    	add    %dh,0x201d(%rax)
 a1d:	00 00                	add    %al,(%rax)
 a1f:	00 08                	add    %cl,(%rax)
 a21:	00 00                	add    %al,(%rax)
 a23:	00 00                	add    %al,(%rax)
 a25:	00 00                	add    %al,(%rax)
 a27:	00 90 15 00 00 00    	add    %dl,0x15(%rax)
 a2d:	00 00                	add    %al,(%rax)
 a2f:	00 b8 1d 20 00 00    	add    %bh,0x201d(%rax)
 a35:	00 00                	add    %al,(%rax)
 a37:	00 08                	add    %cl,(%rax)
 a39:	00 00                	add    %al,(%rax)
 a3b:	00 00                	add    %al,(%rax)
 a3d:	00 00                	add    %al,(%rax)
 a3f:	00 50 16             	add    %dl,0x16(%rax)
 a42:	00 00                	add    %al,(%rax)
 a44:	00 00                	add    %al,(%rax)
 a46:	00 00                	add    %al,(%rax)
 a48:	c0 20 20             	shlb   $0x20,(%rax)
 a4b:	00 00                	add    %al,(%rax)
 a4d:	00 00                	add    %al,(%rax)
 a4f:	00 08                	add    %cl,(%rax)
 a51:	00 00                	add    %al,(%rax)
 a53:	00 00                	add    %al,(%rax)
 a55:	00 00                	add    %al,(%rax)
 a57:	00 c0                	add    %al,%al
 a59:	20 20                	and    %ah,(%rax)
 a5b:	00 00                	add    %al,(%rax)
 a5d:	00 00                	add    %al,(%rax)
 a5f:	00 d0                	add    %dl,%al
 a61:	1f                   	(bad)  
 a62:	20 00                	and    %al,(%rax)
 a64:	00 00                	add    %al,(%rax)
 a66:	00 00                	add    %al,(%rax)
 a68:	06                   	(bad)  
 a69:	00 00                	add    %al,(%rax)
 a6b:	00 02                	add    %al,(%rdx)
	...
 a75:	00 00                	add    %al,(%rax)
 a77:	00 d8                	add    %bl,%al
 a79:	1f                   	(bad)  
 a7a:	20 00                	and    %al,(%rax)
 a7c:	00 00                	add    %al,(%rax)
 a7e:	00 00                	add    %al,(%rax)
 a80:	06                   	(bad)  
 a81:	00 00                	add    %al,(%rax)
 a83:	00 16                	add    %dl,(%rsi)
	...
 a8d:	00 00                	add    %al,(%rax)
 a8f:	00 e0                	add    %ah,%al
 a91:	1f                   	(bad)  
 a92:	20 00                	and    %al,(%rax)
 a94:	00 00                	add    %al,(%rax)
 a96:	00 00                	add    %al,(%rax)
 a98:	06                   	(bad)  
 a99:	00 00                	add    %al,(%rax)
 a9b:	00 18                	add    %bl,(%rax)
	...
 aa5:	00 00                	add    %al,(%rax)
 aa7:	00 e8                	add    %ch,%al
 aa9:	1f                   	(bad)  
 aaa:	20 00                	and    %al,(%rax)
 aac:	00 00                	add    %al,(%rax)
 aae:	00 00                	add    %al,(%rax)
 ab0:	06                   	(bad)  
 ab1:	00 00                	add    %al,(%rax)
 ab3:	00 19                	add    %bl,(%rcx)
	...
 abd:	00 00                	add    %al,(%rax)
 abf:	00 f0                	add    %dh,%al
 ac1:	1f                   	(bad)  
 ac2:	20 00                	and    %al,(%rax)
 ac4:	00 00                	add    %al,(%rax)
 ac6:	00 00                	add    %al,(%rax)
 ac8:	06                   	(bad)  
 ac9:	00 00                	add    %al,(%rax)
 acb:	00 1a                	add    %bl,(%rdx)
	...
 ad5:	00 00                	add    %al,(%rax)
 ad7:	00 f8                	add    %bh,%al
 ad9:	1f                   	(bad)  
 ada:	20 00                	and    %al,(%rax)
 adc:	00 00                	add    %al,(%rax)
 ade:	00 00                	add    %al,(%rax)
 ae0:	06                   	(bad)  
 ae1:	00 00                	add    %al,(%rax)
 ae3:	00 1b                	add    %bl,(%rbx)
	...
 aed:	00 00                	add    %al,(%rax)
 aef:	00 c8                	add    %cl,%al
 af1:	20 20                	and    %ah,(%rax)
 af3:	00 00                	add    %al,(%rax)
 af5:	00 00                	add    %al,(%rax)
 af7:	00 01                	add    %al,(%rcx)
 af9:	00 00                	add    %al,(%rax)
 afb:	00 13                	add    %dl,(%rbx)
	...
 b05:	00 00                	add    %al,(%rax)
 b07:	00 e0                	add    %ah,%al
 b09:	20 20                	and    %ah,(%rax)
 b0b:	00 00                	add    %al,(%rax)
 b0d:	00 00                	add    %al,(%rax)
 b0f:	00 05 00 00 00 1c    	add    %al,0x1c000000(%rip)        # 1c000b15 <_end+0x1bdfe7e5>
	...
 b21:	22 20                	and    (%rax),%ah
 b23:	00 00                	add    %al,(%rax)
 b25:	00 00                	add    %al,(%rax)
 b27:	00 05 00 00 00 1d    	add    %al,0x1d000000(%rip)        # 1d000b2d <_end+0x1cdfe7fd>
	...

Disassembly of section .rela.plt:

0000000000000b38 <.rela.plt>:
 b38:	18 20                	sbb    %ah,(%rax)
 b3a:	20 00                	and    %al,(%rax)
 b3c:	00 00                	add    %al,(%rax)
 b3e:	00 00                	add    %al,(%rax)
 b40:	07                   	(bad)  
 b41:	00 00                	add    %al,(%rax)
 b43:	00 01                	add    %al,(%rcx)
	...
 b4d:	00 00                	add    %al,(%rax)
 b4f:	00 20                	add    %ah,(%rax)
 b51:	20 20                	and    %ah,(%rax)
 b53:	00 00                	add    %al,(%rax)
 b55:	00 00                	add    %al,(%rax)
 b57:	00 07                	add    %al,(%rdi)
 b59:	00 00                	add    %al,(%rax)
 b5b:	00 03                	add    %al,(%rbx)
	...
 b65:	00 00                	add    %al,(%rax)
 b67:	00 28                	add    %ch,(%rax)
 b69:	20 20                	and    %ah,(%rax)
 b6b:	00 00                	add    %al,(%rax)
 b6d:	00 00                	add    %al,(%rax)
 b6f:	00 07                	add    %al,(%rdi)
 b71:	00 00                	add    %al,(%rax)
 b73:	00 04 00             	add    %al,(%rax,%rax,1)
	...
 b7e:	00 00                	add    %al,(%rax)
 b80:	30 20                	xor    %ah,(%rax)
 b82:	20 00                	and    %al,(%rax)
 b84:	00 00                	add    %al,(%rax)
 b86:	00 00                	add    %al,(%rax)
 b88:	07                   	(bad)  
 b89:	00 00                	add    %al,(%rax)
 b8b:	00 05 00 00 00 00    	add    %al,0x0(%rip)        # b91 <_init-0x187>
 b91:	00 00                	add    %al,(%rax)
 b93:	00 00                	add    %al,(%rax)
 b95:	00 00                	add    %al,(%rax)
 b97:	00 38                	add    %bh,(%rax)
 b99:	20 20                	and    %ah,(%rax)
 b9b:	00 00                	add    %al,(%rax)
 b9d:	00 00                	add    %al,(%rax)
 b9f:	00 07                	add    %al,(%rdi)
 ba1:	00 00                	add    %al,(%rax)
 ba3:	00 06                	add    %al,(%rsi)
	...
 bad:	00 00                	add    %al,(%rax)
 baf:	00 40 20             	add    %al,0x20(%rax)
 bb2:	20 00                	and    %al,(%rax)
 bb4:	00 00                	add    %al,(%rax)
 bb6:	00 00                	add    %al,(%rax)
 bb8:	07                   	(bad)  
 bb9:	00 00                	add    %al,(%rax)
 bbb:	00 07                	add    %al,(%rdi)
	...
 bc5:	00 00                	add    %al,(%rax)
 bc7:	00 48 20             	add    %cl,0x20(%rax)
 bca:	20 00                	and    %al,(%rax)
 bcc:	00 00                	add    %al,(%rax)
 bce:	00 00                	add    %al,(%rax)
 bd0:	07                   	(bad)  
 bd1:	00 00                	add    %al,(%rax)
 bd3:	00 08                	add    %cl,(%rax)
	...
 bdd:	00 00                	add    %al,(%rax)
 bdf:	00 50 20             	add    %dl,0x20(%rax)
 be2:	20 00                	and    %al,(%rax)
 be4:	00 00                	add    %al,(%rax)
 be6:	00 00                	add    %al,(%rax)
 be8:	07                   	(bad)  
 be9:	00 00                	add    %al,(%rax)
 beb:	00 09                	add    %cl,(%rcx)
	...
 bf5:	00 00                	add    %al,(%rax)
 bf7:	00 58 20             	add    %bl,0x20(%rax)
 bfa:	20 00                	and    %al,(%rax)
 bfc:	00 00                	add    %al,(%rax)
 bfe:	00 00                	add    %al,(%rax)
 c00:	07                   	(bad)  
 c01:	00 00                	add    %al,(%rax)
 c03:	00 0a                	add    %cl,(%rdx)
	...
 c0d:	00 00                	add    %al,(%rax)
 c0f:	00 60 20             	add    %ah,0x20(%rax)
 c12:	20 00                	and    %al,(%rax)
 c14:	00 00                	add    %al,(%rax)
 c16:	00 00                	add    %al,(%rax)
 c18:	07                   	(bad)  
 c19:	00 00                	add    %al,(%rax)
 c1b:	00 0b                	add    %cl,(%rbx)
	...
 c25:	00 00                	add    %al,(%rax)
 c27:	00 68 20             	add    %ch,0x20(%rax)
 c2a:	20 00                	and    %al,(%rax)
 c2c:	00 00                	add    %al,(%rax)
 c2e:	00 00                	add    %al,(%rax)
 c30:	07                   	(bad)  
 c31:	00 00                	add    %al,(%rax)
 c33:	00 0c 00             	add    %cl,(%rax,%rax,1)
	...
 c3e:	00 00                	add    %al,(%rax)
 c40:	70 20                	jo     c62 <_init-0xb6>
 c42:	20 00                	and    %al,(%rax)
 c44:	00 00                	add    %al,(%rax)
 c46:	00 00                	add    %al,(%rax)
 c48:	07                   	(bad)  
 c49:	00 00                	add    %al,(%rax)
 c4b:	00 0d 00 00 00 00    	add    %cl,0x0(%rip)        # c51 <_init-0xc7>
 c51:	00 00                	add    %al,(%rax)
 c53:	00 00                	add    %al,(%rax)
 c55:	00 00                	add    %al,(%rax)
 c57:	00 78 20             	add    %bh,0x20(%rax)
 c5a:	20 00                	and    %al,(%rax)
 c5c:	00 00                	add    %al,(%rax)
 c5e:	00 00                	add    %al,(%rax)
 c60:	07                   	(bad)  
 c61:	00 00                	add    %al,(%rax)
 c63:	00 0e                	add    %cl,(%rsi)
	...
 c6d:	00 00                	add    %al,(%rax)
 c6f:	00 80 20 20 00 00    	add    %al,0x2020(%rax)
 c75:	00 00                	add    %al,(%rax)
 c77:	00 07                	add    %al,(%rdi)
 c79:	00 00                	add    %al,(%rax)
 c7b:	00 0f                	add    %cl,(%rdi)
	...
 c85:	00 00                	add    %al,(%rax)
 c87:	00 88 20 20 00 00    	add    %cl,0x2020(%rax)
 c8d:	00 00                	add    %al,(%rax)
 c8f:	00 07                	add    %al,(%rdi)
 c91:	00 00                	add    %al,(%rax)
 c93:	00 10                	add    %dl,(%rax)
	...
 c9d:	00 00                	add    %al,(%rax)
 c9f:	00 90 20 20 00 00    	add    %dl,0x2020(%rax)
 ca5:	00 00                	add    %al,(%rax)
 ca7:	00 07                	add    %al,(%rdi)
 ca9:	00 00                	add    %al,(%rax)
 cab:	00 11                	add    %dl,(%rcx)
	...
 cb5:	00 00                	add    %al,(%rax)
 cb7:	00 98 20 20 00 00    	add    %bl,0x2020(%rax)
 cbd:	00 00                	add    %al,(%rax)
 cbf:	00 07                	add    %al,(%rdi)
 cc1:	00 00                	add    %al,(%rax)
 cc3:	00 12                	add    %dl,(%rdx)
	...
 ccd:	00 00                	add    %al,(%rax)
 ccf:	00 a0 20 20 00 00    	add    %ah,0x2020(%rax)
 cd5:	00 00                	add    %al,(%rax)
 cd7:	00 07                	add    %al,(%rdi)
 cd9:	00 00                	add    %al,(%rax)
 cdb:	00 14 00             	add    %dl,(%rax,%rax,1)
	...
 ce6:	00 00                	add    %al,(%rax)
 ce8:	a8 20                	test   $0x20,%al
 cea:	20 00                	and    %al,(%rax)
 cec:	00 00                	add    %al,(%rax)
 cee:	00 00                	add    %al,(%rax)
 cf0:	07                   	(bad)  
 cf1:	00 00                	add    %al,(%rax)
 cf3:	00 15 00 00 00 00    	add    %dl,0x0(%rip)        # cf9 <_init-0x1f>
 cf9:	00 00                	add    %al,(%rax)
 cfb:	00 00                	add    %al,(%rax)
 cfd:	00 00                	add    %al,(%rax)
 cff:	00 b0 20 20 00 00    	add    %dh,0x2020(%rax)
 d05:	00 00                	add    %al,(%rax)
 d07:	00 07                	add    %al,(%rdi)
 d09:	00 00                	add    %al,(%rax)
 d0b:	00 17                	add    %dl,(%rdi)
	...

Disassembly of section .init:

0000000000000d18 <_init>:
 d18:	48 83 ec 08          	sub    $0x8,%rsp
 d1c:	48 8b 05 c5 12 20 00 	mov    0x2012c5(%rip),%rax        # 201fe8 <__gmon_start__>
 d23:	48 85 c0             	test   %rax,%rax
 d26:	74 02                	je     d2a <_init+0x12>
 d28:	ff d0                	callq  *%rax
 d2a:	48 83 c4 08          	add    $0x8,%rsp
 d2e:	c3                   	retq   

Disassembly of section .plt:

0000000000000d30 <.plt>:
 d30:	ff 35 d2 12 20 00    	pushq  0x2012d2(%rip)        # 202008 <_GLOBAL_OFFSET_TABLE_+0x8>
 d36:	ff 25 d4 12 20 00    	jmpq   *0x2012d4(%rip)        # 202010 <_GLOBAL_OFFSET_TABLE_+0x10>
 d3c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000000d40 <_ZNSt8ios_base15sync_with_stdioEb@plt>:
 d40:	ff 25 d2 12 20 00    	jmpq   *0x2012d2(%rip)        # 202018 <_ZNSt8ios_base15sync_with_stdioEb@GLIBCXX_3.4>
 d46:	68 00 00 00 00       	pushq  $0x0
 d4b:	e9 e0 ff ff ff       	jmpq   d30 <.plt>

0000000000000d50 <memset@plt>:
 d50:	ff 25 ca 12 20 00    	jmpq   *0x2012ca(%rip)        # 202020 <memset@GLIBC_2.2.5>
 d56:	68 01 00 00 00       	pushq  $0x1
 d5b:	e9 d0 ff ff ff       	jmpq   d30 <.plt>

0000000000000d60 <_ZNSo5flushEv@plt>:
 d60:	ff 25 c2 12 20 00    	jmpq   *0x2012c2(%rip)        # 202028 <_ZNSo5flushEv@GLIBCXX_3.4>
 d66:	68 02 00 00 00       	pushq  $0x2
 d6b:	e9 c0 ff ff ff       	jmpq   d30 <.plt>

0000000000000d70 <aligned_alloc@plt>:
 d70:	ff 25 ba 12 20 00    	jmpq   *0x2012ba(%rip)        # 202030 <aligned_alloc@GLIBC_2.16>
 d76:	68 03 00 00 00       	pushq  $0x3
 d7b:	e9 b0 ff ff ff       	jmpq   d30 <.plt>

0000000000000d80 <__cxa_atexit@plt>:
 d80:	ff 25 b2 12 20 00    	jmpq   *0x2012b2(%rip)        # 202038 <__cxa_atexit@GLIBC_2.2.5>
 d86:	68 04 00 00 00       	pushq  $0x4
 d8b:	e9 a0 ff ff ff       	jmpq   d30 <.plt>

0000000000000d90 <time@plt>:
 d90:	ff 25 aa 12 20 00    	jmpq   *0x2012aa(%rip)        # 202040 <time@GLIBC_2.2.5>
 d96:	68 05 00 00 00       	pushq  $0x5
 d9b:	e9 90 ff ff ff       	jmpq   d30 <.plt>

0000000000000da0 <_ZNSt13random_device7_M_finiEv@plt>:
 da0:	ff 25 a2 12 20 00    	jmpq   *0x2012a2(%rip)        # 202048 <_ZNSt13random_device7_M_finiEv@GLIBCXX_3.4.18>
 da6:	68 06 00 00 00       	pushq  $0x6
 dab:	e9 80 ff ff ff       	jmpq   d30 <.plt>

0000000000000db0 <_ZdlPv@plt>:
 db0:	ff 25 9a 12 20 00    	jmpq   *0x20129a(%rip)        # 202050 <_ZdlPv@GLIBCXX_3.4>
 db6:	68 07 00 00 00       	pushq  $0x7
 dbb:	e9 70 ff ff ff       	jmpq   d30 <.plt>

0000000000000dc0 <srand@plt>:
 dc0:	ff 25 92 12 20 00    	jmpq   *0x201292(%rip)        # 202058 <srand@GLIBC_2.2.5>
 dc6:	68 08 00 00 00       	pushq  $0x8
 dcb:	e9 60 ff ff ff       	jmpq   d30 <.plt>

0000000000000dd0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>:
 dd0:	ff 25 8a 12 20 00    	jmpq   *0x20128a(%rip)        # 202060 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@GLIBCXX_3.4>
 dd6:	68 09 00 00 00       	pushq  $0x9
 ddb:	e9 50 ff ff ff       	jmpq   d30 <.plt>

0000000000000de0 <_Znwm@plt>:
 de0:	ff 25 82 12 20 00    	jmpq   *0x201282(%rip)        # 202068 <_Znwm@GLIBCXX_3.4>
 de6:	68 0a 00 00 00       	pushq  $0xa
 deb:	e9 40 ff ff ff       	jmpq   d30 <.plt>

0000000000000df0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@plt>:
 df0:	ff 25 7a 12 20 00    	jmpq   *0x20127a(%rip)        # 202070 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@GLIBCXX_3.4>
 df6:	68 0b 00 00 00       	pushq  $0xb
 dfb:	e9 30 ff ff ff       	jmpq   d30 <.plt>

0000000000000e00 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>:
 e00:	ff 25 72 12 20 00    	jmpq   *0x201272(%rip)        # 202078 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@GLIBCXX_3.4.9>
 e06:	68 0c 00 00 00       	pushq  $0xc
 e0b:	e9 20 ff ff ff       	jmpq   d30 <.plt>

0000000000000e10 <_ZNSt13random_device9_M_getvalEv@plt>:
 e10:	ff 25 6a 12 20 00    	jmpq   *0x20126a(%rip)        # 202080 <_ZNSt13random_device9_M_getvalEv@GLIBCXX_3.4.18>
 e16:	68 0d 00 00 00       	pushq  $0xd
 e1b:	e9 10 ff ff ff       	jmpq   d30 <.plt>

0000000000000e20 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>:
 e20:	ff 25 62 12 20 00    	jmpq   *0x201262(%rip)        # 202088 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@GLIBCXX_3.4>
 e26:	68 0e 00 00 00       	pushq  $0xe
 e2b:	e9 00 ff ff ff       	jmpq   d30 <.plt>

0000000000000e30 <_ZNSt8ios_base4InitC1Ev@plt>:
 e30:	ff 25 5a 12 20 00    	jmpq   *0x20125a(%rip)        # 202090 <_ZNSt8ios_base4InitC1Ev@GLIBCXX_3.4>
 e36:	68 0f 00 00 00       	pushq  $0xf
 e3b:	e9 f0 fe ff ff       	jmpq   d30 <.plt>

0000000000000e40 <memmove@plt>:
 e40:	ff 25 52 12 20 00    	jmpq   *0x201252(%rip)        # 202098 <memmove@GLIBC_2.2.5>
 e46:	68 10 00 00 00       	pushq  $0x10
 e4b:	e9 e0 fe ff ff       	jmpq   d30 <.plt>

0000000000000e50 <_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@plt>:
 e50:	ff 25 4a 12 20 00    	jmpq   *0x20124a(%rip)        # 2020a0 <_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@GLIBCXX_3.4.21>
 e56:	68 11 00 00 00       	pushq  $0x11
 e5b:	e9 d0 fe ff ff       	jmpq   d30 <.plt>

0000000000000e60 <_ZNSolsEi@plt>:
 e60:	ff 25 42 12 20 00    	jmpq   *0x201242(%rip)        # 2020a8 <_ZNSolsEi@GLIBCXX_3.4>
 e66:	68 12 00 00 00       	pushq  $0x12
 e6b:	e9 c0 fe ff ff       	jmpq   d30 <.plt>

0000000000000e70 <_Unwind_Resume@plt>:
 e70:	ff 25 3a 12 20 00    	jmpq   *0x20123a(%rip)        # 2020b0 <_Unwind_Resume@GCC_3.0>
 e76:	68 13 00 00 00       	pushq  $0x13
 e7b:	e9 b0 fe ff ff       	jmpq   d30 <.plt>

Disassembly of section .plt.got:

0000000000000e80 <__cxa_finalize@plt>:
 e80:	ff 25 4a 11 20 00    	jmpq   *0x20114a(%rip)        # 201fd0 <__cxa_finalize@GLIBC_2.2.5>
 e86:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000000e90 <_GLOBAL__sub_I__Z8randinitv>:
     e90:	48 83 ec 08          	sub    $0x8,%rsp
     e94:	48 8d 3d 85 14 20 00 	lea    0x201485(%rip),%rdi        # 202320 <_ZStL8__ioinit>
     e9b:	e8 90 ff ff ff       	callq  e30 <_ZNSt8ios_base4InitC1Ev@plt>
     ea0:	48 8b 3d 51 11 20 00 	mov    0x201151(%rip),%rdi        # 201ff8 <_ZNSt8ios_base4InitD1Ev@GLIBCXX_3.4>
     ea7:	48 8d 15 12 12 20 00 	lea    0x201212(%rip),%rdx        # 2020c0 <__dso_handle>
     eae:	48 8d 35 6b 14 20 00 	lea    0x20146b(%rip),%rsi        # 202320 <_ZStL8__ioinit>
     eb5:	e8 c6 fe ff ff       	callq  d80 <__cxa_atexit@plt>
     eba:	48 c7 05 63 14 20 00 	movq   $0x1,0x201463(%rip)        # 202328 <_ZL9generator>
     ec1:	01 00 00 00 
     ec5:	48 83 c4 08          	add    $0x8,%rsp
     ec9:	c3                   	retq   
     eca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000000ed0 <main>:
     ed0:	55                   	push   %rbp
     ed1:	31 ff                	xor    %edi,%edi
     ed3:	48 89 e5             	mov    %rsp,%rbp
     ed6:	41 57                	push   %r15
     ed8:	41 56                	push   %r14
     eda:	41 55                	push   %r13
     edc:	41 54                	push   %r12
     ede:	53                   	push   %rbx
     edf:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
     ee3:	48 81 ec e0 13 00 00 	sub    $0x13e0,%rsp
     eea:	e8 51 fe ff ff       	callq  d40 <_ZNSt8ios_base15sync_with_stdioEb@plt>
     eef:	48 8d 5c 24 30       	lea    0x30(%rsp),%rbx
     ef4:	c7 44 24 40 64 65 66 	movl   $0x61666564,0x40(%rsp)
     efb:	61 
     efc:	48 8d 43 10          	lea    0x10(%rbx),%rax
     f00:	c6 43 16 74          	movb   $0x74,0x16(%rbx)
     f04:	48 89 de             	mov    %rbx,%rsi
     f07:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
     f0c:	b8 75 6c 00 00       	mov    $0x6c75,%eax
     f11:	66 89 43 14          	mov    %ax,0x14(%rbx)
     f15:	48 8d 44 24 50       	lea    0x50(%rsp),%rax
     f1a:	48 89 c7             	mov    %rax,%rdi
     f1d:	c6 44 24 47 00       	movb   $0x0,0x47(%rsp)
     f22:	48 c7 05 bb 13 20 00 	movq   $0x0,0x2013bb(%rip)        # 2022e8 <_ZSt3cin@@GLIBCXX_3.4+0xe8>
     f29:	00 00 00 00 
     f2d:	48 c7 44 24 38 07 00 	movq   $0x7,0x38(%rsp)
     f34:	00 00 
     f36:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
     f3b:	e8 10 ff ff ff       	callq  e50 <_ZNSt13random_device7_M_initERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE@plt>
     f40:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
     f45:	48 83 c3 10          	add    $0x10,%rbx
     f49:	48 39 df             	cmp    %rbx,%rdi
     f4c:	74 05                	je     f53 <main+0x83>
     f4e:	e8 5d fe ff ff       	callq  db0 <_ZdlPv@plt>
     f53:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
     f58:	48 89 df             	mov    %rbx,%rdi
     f5b:	e8 b0 fe ff ff       	callq  e10 <_ZNSt13random_device9_M_getvalEv@plt>
     f60:	48 ba 05 00 00 00 02 	movabs $0x200000005,%rdx
     f67:	00 00 00 
     f6a:	89 c1                	mov    %eax,%ecx
     f6c:	48 89 c8             	mov    %rcx,%rax
     f6f:	48 f7 e2             	mul    %rdx
     f72:	48 89 c8             	mov    %rcx,%rax
     f75:	48 29 d0             	sub    %rdx,%rax
     f78:	48 d1 e8             	shr    %rax
     f7b:	48 01 d0             	add    %rdx,%rax
     f7e:	48 c1 e8 1e          	shr    $0x1e,%rax
     f82:	48 89 c2             	mov    %rax,%rdx
     f85:	48 c1 e2 1f          	shl    $0x1f,%rdx
     f89:	48 29 c2             	sub    %rax,%rdx
     f8c:	48 29 d1             	sub    %rdx,%rcx
     f8f:	ba 01 00 00 00       	mov    $0x1,%edx
     f94:	48 89 c8             	mov    %rcx,%rax
     f97:	48 0f 44 c2          	cmove  %rdx,%rax
     f9b:	31 ff                	xor    %edi,%edi
     f9d:	48 89 05 84 13 20 00 	mov    %rax,0x201384(%rip)        # 202328 <_ZL9generator>
     fa4:	e8 e7 fd ff ff       	callq  d90 <time@plt>
     fa9:	89 c7                	mov    %eax,%edi
     fab:	e8 10 fe ff ff       	callq  dc0 <srand@plt>
     fb0:	48 89 df             	mov    %rbx,%rdi
     fb3:	e8 e8 fd ff ff       	callq  da0 <_ZNSt13random_device7_M_finiEv@plt>
     fb8:	be 00 00 40 00       	mov    $0x400000,%esi
     fbd:	bf 80 00 00 00       	mov    $0x80,%edi
     fc2:	e8 a9 fd ff ff       	callq  d70 <aligned_alloc@plt>
     fc7:	be 00 00 40 00       	mov    $0x400000,%esi
     fcc:	bf 80 00 00 00       	mov    $0x80,%edi
     fd1:	48 89 c3             	mov    %rax,%rbx
     fd4:	e8 97 fd ff ff       	callq  d70 <aligned_alloc@plt>
     fd9:	48 8d 35 58 09 00 00 	lea    0x958(%rip),%rsi        # 1938 <_IO_stdin_used+0x58>
     fe0:	48 8d 3d f9 10 20 00 	lea    0x2010f9(%rip),%rdi        # 2020e0 <_ZSt4cout@@GLIBCXX_3.4>
     fe7:	49 89 c5             	mov    %rax,%r13
     fea:	e8 e1 fd ff ff       	callq  dd0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
     fef:	c5 fd 6f 15 e9 09 00 	vmovdqa 0x9e9(%rip),%ymm2        # 19e0 <_IO_stdin_used+0x100>
     ff6:	00 
     ff7:	c5 fd 6f 25 01 0a 00 	vmovdqa 0xa01(%rip),%ymm4        # 1a00 <_IO_stdin_used+0x120>
     ffe:	00 
     fff:	48 89 d8             	mov    %rbx,%rax
    1002:	c5 fd 28 1d 36 0a 00 	vmovapd 0xa36(%rip),%ymm3        # 1a40 <_IO_stdin_used+0x160>
    1009:	00 
    100a:	48 8d 93 00 00 40 00 	lea    0x400000(%rbx),%rdx
    1011:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1016:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    101d:	00 00 00 
    1020:	c5 ed fe 05 f8 09 00 	vpaddd 0x9f8(%rip),%ymm2,%ymm0        # 1a20 <_IO_stdin_used+0x140>
    1027:	00 
    1028:	c4 e3 7d 39 d5 01    	vextracti128 $0x1,%ymm2,%xmm5
    102e:	48 83 c0 20          	add    $0x20,%rax
    1032:	c5 fe e6 ed          	vcvtdq2pd %xmm5,%ymm5
    1036:	c5 d5 59 eb          	vmulpd %ymm3,%ymm5,%ymm5
    103a:	c5 fe e6 c8          	vcvtdq2pd %xmm0,%ymm1
    103e:	c5 f5 59 cb          	vmulpd %ymm3,%ymm1,%ymm1
    1042:	c4 e3 7d 39 c0 01    	vextracti128 $0x1,%ymm0,%xmm0
    1048:	c5 fe e6 c0          	vcvtdq2pd %xmm0,%ymm0
    104c:	c5 fd 59 c3          	vmulpd %ymm3,%ymm0,%ymm0
    1050:	c5 fd e6 ed          	vcvttpd2dq %ymm5,%xmm5
    1054:	c5 fd e6 c9          	vcvttpd2dq %ymm1,%xmm1
    1058:	c5 fd e6 c0          	vcvttpd2dq %ymm0,%xmm0
    105c:	c4 e3 75 38 c0 01    	vinserti128 $0x1,%xmm0,%ymm1,%ymm0
    1062:	c5 fe e6 ca          	vcvtdq2pd %xmm2,%ymm1
    1066:	c5 ed fe d4          	vpaddd %ymm4,%ymm2,%ymm2
    106a:	c5 f5 59 cb          	vmulpd %ymm3,%ymm1,%ymm1
    106e:	c5 fd e6 c9          	vcvttpd2dq %ymm1,%xmm1
    1072:	c4 e3 75 38 cd 01    	vinserti128 $0x1,%xmm5,%ymm1,%ymm1
    1078:	c5 fd fa c1          	vpsubd %ymm1,%ymm0,%ymm0
    107c:	c5 fe 7f 40 e0       	vmovdqu %ymm0,-0x20(%rax)
    1081:	48 39 d0             	cmp    %rdx,%rax
    1084:	75 9a                	jne    1020 <main+0x150>
    1086:	48 8d 35 89 08 00 00 	lea    0x889(%rip),%rsi        # 1916 <_IO_stdin_used+0x36>
    108d:	48 8d 3d 4c 10 20 00 	lea    0x20104c(%rip),%rdi        # 2020e0 <_ZSt4cout@@GLIBCXX_3.4>
    1094:	c5 f8 77             	vzeroupper 
    1097:	e8 34 fd ff ff       	callq  dd0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    109c:	45 31 e4             	xor    %r12d,%r12d
    109f:	45 31 f6             	xor    %r14d,%r14d
    10a2:	48 8d 35 bf 08 00 00 	lea    0x8bf(%rip),%rsi        # 1968 <_IO_stdin_used+0x88>
    10a9:	48 8d 3d 30 10 20 00 	lea    0x201030(%rip),%rdi        # 2020e0 <_ZSt4cout@@GLIBCXX_3.4>
    10b0:	49 bf 05 00 00 00 02 	movabs $0x200000005,%r15
    10b7:	00 00 00 
    10ba:	e8 11 fd ff ff       	callq  dd0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    10bf:	31 c9                	xor    %ecx,%ecx
    10c1:	48 c7 44 24 20 00 00 	movq   $0x0,0x20(%rsp)
    10c8:	00 00 
    10ca:	eb 14                	jmp    10e0 <main+0x210>
    10cc:	0f 1f 40 00          	nopl   0x0(%rax)
    10d0:	49 ff c4             	inc    %r12
    10d3:	49 81 fc 00 00 10 00 	cmp    $0x100000,%r12
    10da:	0f 84 a7 00 00 00    	je     1187 <main+0x2b7>
    10e0:	42 83 3c a3 01       	cmpl   $0x1,(%rbx,%r12,4)
    10e5:	44 89 e7             	mov    %r12d,%edi
    10e8:	75 e6                	jne    10d0 <main+0x200>
    10ea:	42 c7 04 a3 00 00 00 	movl   $0x0,(%rbx,%r12,4)
    10f1:	00 
    10f2:	48 8b 15 2f 12 20 00 	mov    0x20122f(%rip),%rdx        # 202328 <_ZL9generator>
    10f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1100:	48 69 f2 a7 41 00 00 	imul   $0x41a7,%rdx,%rsi
    1107:	48 89 f0             	mov    %rsi,%rax
    110a:	49 f7 e7             	mul    %r15
    110d:	48 89 f0             	mov    %rsi,%rax
    1110:	48 29 d0             	sub    %rdx,%rax
    1113:	48 d1 e8             	shr    %rax
    1116:	48 01 c2             	add    %rax,%rdx
    1119:	48 c1 ea 1e          	shr    $0x1e,%rdx
    111d:	48 89 d0             	mov    %rdx,%rax
    1120:	48 c1 e0 1f          	shl    $0x1f,%rax
    1124:	48 29 d0             	sub    %rdx,%rax
    1127:	48 29 c6             	sub    %rax,%rsi
    112a:	48 8d 46 ff          	lea    -0x1(%rsi),%rax
    112e:	48 89 f2             	mov    %rsi,%rdx
    1131:	48 3d fb ff ff 7f    	cmp    $0x7ffffffb,%rax
    1137:	77 c7                	ja     1100 <main+0x230>
    1139:	48 d1 e8             	shr    %rax
    113c:	48 89 35 e5 11 20 00 	mov    %rsi,0x2011e5(%rip)        # 202328 <_ZL9generator>
    1143:	48 89 c2             	mov    %rax,%rdx
    1146:	48 b8 21 00 00 00 04 	movabs $0x8000000400000021,%rax
    114d:	00 00 80 
    1150:	48 f7 e2             	mul    %rdx
    1153:	48 c1 ea 1c          	shr    $0x1c,%rdx
    1157:	8d 84 57 ff ff 0f 00 	lea    0xfffff(%rdi,%rdx,2),%eax
    115e:	25 ff ff 0f 00       	and    $0xfffff,%eax
    1163:	89 44 24 28          	mov    %eax,0x28(%rsp)
    1167:	4c 39 f1             	cmp    %r14,%rcx
    116a:	0f 84 a0 02 00 00    	je     1410 <main+0x540>
    1170:	49 ff c4             	inc    %r12
    1173:	41 89 06             	mov    %eax,(%r14)
    1176:	49 83 c6 04          	add    $0x4,%r14
    117a:	49 81 fc 00 00 10 00 	cmp    $0x100000,%r12
    1181:	0f 85 59 ff ff ff    	jne    10e0 <main+0x210>
    1187:	4c 2b 74 24 20       	sub    0x20(%rsp),%r14
    118c:	31 d2                	xor    %edx,%edx
    118e:	31 c0                	xor    %eax,%eax
    1190:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    1195:	49 c1 fe 02          	sar    $0x2,%r14
    1199:	74 17                	je     11b2 <main+0x2e2>
    119b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    11a0:	48 63 14 91          	movslq (%rcx,%rdx,4),%rdx
    11a4:	ff 04 93             	incl   (%rbx,%rdx,4)
    11a7:	8d 50 01             	lea    0x1(%rax),%edx
    11aa:	48 89 d0             	mov    %rdx,%rax
    11ad:	4c 39 f2             	cmp    %r14,%rdx
    11b0:	72 ee                	jb     11a0 <main+0x2d0>
    11b2:	48 83 7c 24 20 00    	cmpq   $0x0,0x20(%rsp)
    11b8:	74 0a                	je     11c4 <main+0x2f4>
    11ba:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    11bf:	e8 ec fb ff ff       	callq  db0 <_ZdlPv@plt>
    11c4:	48 8d 35 4b 07 00 00 	lea    0x74b(%rip),%rsi        # 1916 <_IO_stdin_used+0x36>
    11cb:	48 8d 3d 0e 0f 20 00 	lea    0x200f0e(%rip),%rdi        # 2020e0 <_ZSt4cout@@GLIBCXX_3.4>
    11d2:	49 bc 21 00 00 00 04 	movabs $0x8000000400000021,%r12
    11d9:	00 00 80 
    11dc:	e8 ef fb ff ff       	callq  dd0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    11e1:	48 8d 35 b8 07 00 00 	lea    0x7b8(%rip),%rsi        # 19a0 <_IO_stdin_used+0xc0>
    11e8:	48 8d 3d f1 0e 20 00 	lea    0x200ef1(%rip),%rdi        # 2020e0 <_ZSt4cout@@GLIBCXX_3.4>
    11ef:	e8 dc fb ff ff       	callq  dd0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    11f4:	48 8d 3d e5 0e 20 00 	lea    0x200ee5(%rip),%rdi        # 2020e0 <_ZSt4cout@@GLIBCXX_3.4>
    11fb:	e8 60 fb ff ff       	callq  d60 <_ZNSo5flushEv@plt>
    1200:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    1205:	ba 10 00 00 00       	mov    $0x10,%edx
    120a:	48 8d 35 0c 07 00 00 	lea    0x70c(%rip),%rsi        # 191d <_IO_stdin_used+0x3d>
    1211:	e8 da fb ff ff       	callq  df0 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEEC1EPKcSt13_Ios_Openmode@plt>
    1216:	c7 44 24 28 00 00 00 	movl   $0x0,0x28(%rsp)
    121d:	00 
    121e:	66 90                	xchg   %ax,%ax
    1220:	c7 03 56 34 12 00    	movl   $0x123456,(%rbx)
    1226:	31 f6                	xor    %esi,%esi
    1228:	ba 00 00 40 00       	mov    $0x400000,%edx
    122d:	4c 89 ef             	mov    %r13,%rdi
    1230:	e8 1b fb ff ff       	callq  d50 <memset@plt>
    1235:	be 01 00 00 00       	mov    $0x1,%esi
    123a:	eb 14                	jmp    1250 <main+0x380>
    123c:	0f 1f 40 00          	nopl   0x0(%rax)
    1240:	48 ff c6             	inc    %rsi
    1243:	48 81 fe 01 00 10 00 	cmp    $0x100001,%rsi
    124a:	0f 84 c3 00 00 00    	je     1313 <main+0x443>
    1250:	44 8b 44 b3 fc       	mov    -0x4(%rbx,%rsi,4),%r8d
    1255:	89 f7                	mov    %esi,%edi
    1257:	41 89 f2             	mov    %esi,%r10d
    125a:	41 83 f8 01          	cmp    $0x1,%r8d
    125e:	7e e0                	jle    1240 <main+0x370>
    1260:	48 8b 0d c1 10 20 00 	mov    0x2010c1(%rip),%rcx        # 202328 <_ZL9generator>
    1267:	4c 8d 35 ba 10 20 00 	lea    0x2010ba(%rip),%r14        # 202328 <_ZL9generator>
    126e:	45 31 c9             	xor    %r9d,%r9d
    1271:	45 31 db             	xor    %r11d,%r11d
    1274:	66 90                	xchg   %ax,%ax
    1276:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    127d:	00 00 00 
    1280:	4c 69 f9 a7 41 00 00 	imul   $0x41a7,%rcx,%r15
    1287:	48 b8 05 00 00 00 02 	movabs $0x200000005,%rax
    128e:	00 00 00 
    1291:	49 f7 e7             	mul    %r15
    1294:	4c 89 f9             	mov    %r15,%rcx
    1297:	48 29 d1             	sub    %rdx,%rcx
    129a:	48 d1 e9             	shr    %rcx
    129d:	48 01 ca             	add    %rcx,%rdx
    12a0:	48 89 d1             	mov    %rdx,%rcx
    12a3:	48 c1 e9 1e          	shr    $0x1e,%rcx
    12a7:	48 89 c8             	mov    %rcx,%rax
    12aa:	48 c1 e0 1f          	shl    $0x1f,%rax
    12ae:	48 29 c8             	sub    %rcx,%rax
    12b1:	4c 89 f9             	mov    %r15,%rcx
    12b4:	48 29 c1             	sub    %rax,%rcx
    12b7:	48 8d 41 ff          	lea    -0x1(%rcx),%rax
    12bb:	48 3d fb ff ff 7f    	cmp    $0x7ffffffb,%rax
    12c1:	77 bd                	ja     1280 <main+0x3b0>
    12c3:	48 d1 e8             	shr    %rax
    12c6:	41 ff c3             	inc    %r11d
    12c9:	49 89 0e             	mov    %rcx,(%r14)
    12cc:	49 f7 e4             	mul    %r12
    12cf:	48 c1 ea 1c          	shr    $0x1c,%rdx
    12d3:	41 01 d1             	add    %edx,%r9d
    12d6:	45 39 d8             	cmp    %r11d,%r8d
    12d9:	75 a5                	jne    1280 <main+0x3b0>
    12db:	81 c7 fe ff 0f 00    	add    $0xffffe,%edi
    12e1:	41 81 e2 ff ff 0f 00 	and    $0xfffff,%r10d
    12e8:	c7 44 b3 fc 00 00 00 	movl   $0x0,-0x4(%rbx,%rsi,4)
    12ef:	00 
    12f0:	48 ff c6             	inc    %rsi
    12f3:	81 e7 ff ff 0f 00    	and    $0xfffff,%edi
    12f9:	47 01 4c 95 00       	add    %r9d,0x0(%r13,%r10,4)
    12fe:	45 29 c8             	sub    %r9d,%r8d
    1301:	45 01 44 bd 00       	add    %r8d,0x0(%r13,%rdi,4)
    1306:	48 81 fe 01 00 10 00 	cmp    $0x100001,%rsi
    130d:	0f 85 3d ff ff ff    	jne    1250 <main+0x380>
    1313:	c7 03 77 77 00 00    	movl   $0x7777,(%rbx)
    1319:	31 c0                	xor    %eax,%eax
    131b:	c5 f1 ef c9          	vpxor  %xmm1,%xmm1,%xmm1
    131f:	90                   	nop
    1320:	c4 c1 7d 6f 74 05 00 	vmovdqa 0x0(%r13,%rax,1),%ymm6
    1327:	c5 cd fe 04 03       	vpaddd (%rbx,%rax,1),%ymm6,%ymm0
    132c:	c5 fd 7f 04 03       	vmovdqa %ymm0,(%rbx,%rax,1)
    1331:	48 83 c0 20          	add    $0x20,%rax
    1335:	c5 fd 66 05 e3 06 00 	vpcmpgtd 0x6e3(%rip),%ymm0,%ymm0        # 1a20 <_IO_stdin_used+0x140>
    133c:	00 
    133d:	c5 f5 fa c8          	vpsubd %ymm0,%ymm1,%ymm1
    1341:	48 3d 00 00 40 00    	cmp    $0x400000,%rax
    1347:	75 d7                	jne    1320 <main+0x450>
    1349:	c4 e3 7d 39 c8 01    	vextracti128 $0x1,%ymm1,%xmm0
    134f:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    1354:	c5 f9 fe c9          	vpaddd %xmm1,%xmm0,%xmm1
    1358:	c5 f9 73 d9 08       	vpsrldq $0x8,%xmm1,%xmm0
    135d:	c5 f1 fe c8          	vpaddd %xmm0,%xmm1,%xmm1
    1361:	c5 f9 73 d9 04       	vpsrldq $0x4,%xmm1,%xmm0
    1366:	c5 f1 fe c8          	vpaddd %xmm0,%xmm1,%xmm1
    136a:	c4 c1 79 7e ce       	vmovd  %xmm1,%r14d
    136f:	c5 f9 7e ce          	vmovd  %xmm1,%esi
    1373:	c5 f8 77             	vzeroupper 
    1376:	e8 e5 fa ff ff       	callq  e60 <_ZNSolsEi@plt>
    137b:	ba 01 00 00 00       	mov    $0x1,%edx
    1380:	48 8d 35 70 05 00 00 	lea    0x570(%rip),%rsi        # 18f7 <_IO_stdin_used+0x17>
    1387:	48 89 c7             	mov    %rax,%rdi
    138a:	e8 71 fa ff ff       	callq  e00 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    138f:	ff 44 24 28          	incl   0x28(%rsp)
    1393:	8b 44 24 28          	mov    0x28(%rsp),%eax
    1397:	45 85 f6             	test   %r14d,%r14d
    139a:	7e 0b                	jle    13a7 <main+0x4d7>
    139c:	3d 0f 27 00 00       	cmp    $0x270f,%eax
    13a1:	0f 8e 79 fe ff ff    	jle    1220 <main+0x350>
    13a7:	ba 07 00 00 00       	mov    $0x7,%edx
    13ac:	48 8d 35 77 05 00 00 	lea    0x577(%rip),%rsi        # 192a <_IO_stdin_used+0x4a>
    13b3:	48 8d 3d 26 0d 20 00 	lea    0x200d26(%rip),%rdi        # 2020e0 <_ZSt4cout@@GLIBCXX_3.4>
    13ba:	e8 41 fa ff ff       	callq  e00 <_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l@plt>
    13bf:	45 85 f6             	test   %r14d,%r14d
    13c2:	48 8d 35 1b 05 00 00 	lea    0x51b(%rip),%rsi        # 18e4 <_IO_stdin_used+0x4>
    13c9:	48 8d 05 29 05 00 00 	lea    0x529(%rip),%rax        # 18f9 <_IO_stdin_used+0x19>
    13d0:	48 0f 4e f0          	cmovle %rax,%rsi
    13d4:	48 8d 3d 05 0d 20 00 	lea    0x200d05(%rip),%rdi        # 2020e0 <_ZSt4cout@@GLIBCXX_3.4>
    13db:	e8 f0 f9 ff ff       	callq  dd0 <_ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc@plt>
    13e0:	48 8d 3d f9 0c 20 00 	lea    0x200cf9(%rip),%rdi        # 2020e0 <_ZSt4cout@@GLIBCXX_3.4>
    13e7:	e8 74 f9 ff ff       	callq  d60 <_ZNSo5flushEv@plt>
    13ec:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    13f1:	e8 2a fa ff ff       	callq  e20 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    13f6:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
    13fa:	31 c0                	xor    %eax,%eax
    13fc:	5b                   	pop    %rbx
    13fd:	41 5c                	pop    %r12
    13ff:	41 5d                	pop    %r13
    1401:	41 5e                	pop    %r14
    1403:	41 5f                	pop    %r15
    1405:	5d                   	pop    %rbp
    1406:	c3                   	retq   
    1407:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    140e:	00 00 
    1410:	4c 89 f0             	mov    %r14,%rax
    1413:	48 2b 44 24 20       	sub    0x20(%rsp),%rax
    1418:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    141d:	48 c1 f8 02          	sar    $0x2,%rax
    1421:	0f 84 b9 00 00 00    	je     14e0 <main+0x610>
    1427:	48 c7 44 24 10 fc ff 	movq   $0xfffffffffffffffc,0x10(%rsp)
    142e:	ff ff 
    1430:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    1434:	48 39 d0             	cmp    %rdx,%rax
    1437:	76 77                	jbe    14b0 <main+0x5e0>
    1439:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    143e:	e8 9d f9 ff ff       	callq  de0 <_Znwm@plt>
    1443:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    1448:	49 89 c0             	mov    %rax,%r8
    144b:	48 01 c1             	add    %rax,%rcx
    144e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    1453:	8b 7c 24 28          	mov    0x28(%rsp),%edi
    1457:	48 8b 74 24 20       	mov    0x20(%rsp),%rsi
    145c:	41 89 3c 00          	mov    %edi,(%r8,%rax,1)
    1460:	4c 39 f6             	cmp    %r14,%rsi
    1463:	74 6b                	je     14d0 <main+0x600>
    1465:	4c 89 c7             	mov    %r8,%rdi
    1468:	48 89 c2             	mov    %rax,%rdx
    146b:	48 89 4c 24 28       	mov    %rcx,0x28(%rsp)
    1470:	49 89 c6             	mov    %rax,%r14
    1473:	e8 c8 f9 ff ff       	callq  e40 <memmove@plt>
    1478:	48 8b 4c 24 28       	mov    0x28(%rsp),%rcx
    147d:	49 89 c0             	mov    %rax,%r8
    1480:	4f 8d 74 30 04       	lea    0x4(%r8,%r14,1),%r14
    1485:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    148a:	48 89 4c 24 10       	mov    %rcx,0x10(%rsp)
    148f:	4c 89 44 24 28       	mov    %r8,0x28(%rsp)
    1494:	e8 17 f9 ff ff       	callq  db0 <_ZdlPv@plt>
    1499:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    149e:	4c 8b 44 24 28       	mov    0x28(%rsp),%r8
    14a3:	4c 89 44 24 20       	mov    %r8,0x20(%rsp)
    14a8:	e9 23 fc ff ff       	jmpq   10d0 <main+0x200>
    14ad:	0f 1f 00             	nopl   (%rax)
    14b0:	48 bf ff ff ff ff ff 	movabs $0x3fffffffffffffff,%rdi
    14b7:	ff ff 3f 
    14ba:	48 39 fa             	cmp    %rdi,%rdx
    14bd:	77 2f                	ja     14ee <main+0x61e>
    14bf:	48 85 d2             	test   %rdx,%rdx
    14c2:	75 38                	jne    14fc <main+0x62c>
    14c4:	31 c9                	xor    %ecx,%ecx
    14c6:	45 31 c0             	xor    %r8d,%r8d
    14c9:	eb 83                	jmp    144e <main+0x57e>
    14cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    14d0:	48 83 7c 24 20 00    	cmpq   $0x0,0x20(%rsp)
    14d6:	4d 8d 74 00 04       	lea    0x4(%r8,%rax,1),%r14
    14db:	74 c6                	je     14a3 <main+0x5d3>
    14dd:	eb a6                	jmp    1485 <main+0x5b5>
    14df:	90                   	nop
    14e0:	48 c7 44 24 10 04 00 	movq   $0x4,0x10(%rsp)
    14e7:	00 00 
    14e9:	e9 4b ff ff ff       	jmpq   1439 <main+0x569>
    14ee:	48 c7 44 24 10 fc ff 	movq   $0xfffffffffffffffc,0x10(%rsp)
    14f5:	ff ff 
    14f7:	e9 3d ff ff ff       	jmpq   1439 <main+0x569>
    14fc:	48 c1 e0 03          	shl    $0x3,%rax
    1500:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    1505:	e9 2f ff ff ff       	jmpq   1439 <main+0x569>
    150a:	48 89 c3             	mov    %rax,%rbx
    150d:	eb 05                	jmp    1514 <main+0x644>
    150f:	48 89 c3             	mov    %rax,%rbx
    1512:	eb 1d                	jmp    1531 <main+0x661>
    1514:	48 83 7c 24 20 00    	cmpq   $0x0,0x20(%rsp)
    151a:	74 2a                	je     1546 <main+0x676>
    151c:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    1521:	c5 f8 77             	vzeroupper 
    1524:	e8 87 f8 ff ff       	callq  db0 <_ZdlPv@plt>
    1529:	48 89 df             	mov    %rbx,%rdi
    152c:	e8 3f f9 ff ff       	callq  e70 <_Unwind_Resume@plt>
    1531:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    1536:	c5 f8 77             	vzeroupper 
    1539:	e8 62 f8 ff ff       	callq  da0 <_ZNSt13random_device7_M_finiEv@plt>
    153e:	48 89 df             	mov    %rbx,%rdi
    1541:	e8 2a f9 ff ff       	callq  e70 <_Unwind_Resume@plt>
    1546:	c5 f8 77             	vzeroupper 
    1549:	eb de                	jmp    1529 <main+0x659>
    154b:	49 89 c4             	mov    %rax,%r12
    154e:	eb 05                	jmp    1555 <main+0x685>
    1550:	48 89 c3             	mov    %rax,%rbx
    1553:	eb 1e                	jmp    1573 <main+0x6a3>
    1555:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    155a:	48 83 c3 10          	add    $0x10,%rbx
    155e:	48 39 df             	cmp    %rbx,%rdi
    1561:	74 1f                	je     1582 <main+0x6b2>
    1563:	c5 f8 77             	vzeroupper 
    1566:	e8 45 f8 ff ff       	callq  db0 <_ZdlPv@plt>
    156b:	4c 89 e7             	mov    %r12,%rdi
    156e:	e8 fd f8 ff ff       	callq  e70 <_Unwind_Resume@plt>
    1573:	48 8b 7c 24 18       	mov    0x18(%rsp),%rdi
    1578:	c5 f8 77             	vzeroupper 
    157b:	e8 a0 f8 ff ff       	callq  e20 <_ZNSt14basic_ofstreamIcSt11char_traitsIcEED1Ev@plt>
    1580:	eb a7                	jmp    1529 <main+0x659>
    1582:	c5 f8 77             	vzeroupper 
    1585:	eb e4                	jmp    156b <main+0x69b>
    1587:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    158e:	00 00 

0000000000001590 <set_fast_math>:
    1590:	0f ae 5c 24 fc       	stmxcsr -0x4(%rsp)
    1595:	81 4c 24 fc 40 80 00 	orl    $0x8040,-0x4(%rsp)
    159c:	00 
    159d:	0f ae 54 24 fc       	ldmxcsr -0x4(%rsp)
    15a2:	c3                   	retq   
    15a3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    15aa:	00 00 00 
    15ad:	0f 1f 00             	nopl   (%rax)

00000000000015b0 <_start>:
    15b0:	31 ed                	xor    %ebp,%ebp
    15b2:	49 89 d1             	mov    %rdx,%r9
    15b5:	5e                   	pop    %rsi
    15b6:	48 89 e2             	mov    %rsp,%rdx
    15b9:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    15bd:	50                   	push   %rax
    15be:	54                   	push   %rsp
    15bf:	4c 8d 05 fa 02 00 00 	lea    0x2fa(%rip),%r8        # 18c0 <__libc_csu_fini>
    15c6:	48 8d 0d 83 02 00 00 	lea    0x283(%rip),%rcx        # 1850 <__libc_csu_init>
    15cd:	48 8d 3d fc f8 ff ff 	lea    -0x704(%rip),%rdi        # ed0 <main>
    15d4:	ff 15 06 0a 20 00    	callq  *0x200a06(%rip)        # 201fe0 <__libc_start_main@GLIBC_2.2.5>
    15da:	f4                   	hlt    
    15db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000015e0 <deregister_tm_clones>:
    15e0:	48 8d 3d e9 0a 20 00 	lea    0x200ae9(%rip),%rdi        # 2020d0 <__TMC_END__>
    15e7:	48 8d 05 e2 0a 20 00 	lea    0x200ae2(%rip),%rax        # 2020d0 <__TMC_END__>
    15ee:	48 39 f8             	cmp    %rdi,%rax
    15f1:	74 15                	je     1608 <deregister_tm_clones+0x28>
    15f3:	48 8b 05 de 09 20 00 	mov    0x2009de(%rip),%rax        # 201fd8 <_ITM_deregisterTMCloneTable>
    15fa:	48 85 c0             	test   %rax,%rax
    15fd:	74 09                	je     1608 <deregister_tm_clones+0x28>
    15ff:	ff e0                	jmpq   *%rax
    1601:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1608:	c3                   	retq   
    1609:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001610 <register_tm_clones>:
    1610:	48 8d 3d b9 0a 20 00 	lea    0x200ab9(%rip),%rdi        # 2020d0 <__TMC_END__>
    1617:	48 8d 35 b2 0a 20 00 	lea    0x200ab2(%rip),%rsi        # 2020d0 <__TMC_END__>
    161e:	48 29 fe             	sub    %rdi,%rsi
    1621:	48 c1 fe 03          	sar    $0x3,%rsi
    1625:	48 89 f0             	mov    %rsi,%rax
    1628:	48 c1 e8 3f          	shr    $0x3f,%rax
    162c:	48 01 c6             	add    %rax,%rsi
    162f:	48 d1 fe             	sar    %rsi
    1632:	74 14                	je     1648 <register_tm_clones+0x38>
    1634:	48 8b 05 b5 09 20 00 	mov    0x2009b5(%rip),%rax        # 201ff0 <_ITM_registerTMCloneTable>
    163b:	48 85 c0             	test   %rax,%rax
    163e:	74 08                	je     1648 <register_tm_clones+0x38>
    1640:	ff e0                	jmpq   *%rax
    1642:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1648:	c3                   	retq   
    1649:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001650 <__do_global_dtors_aux>:
    1650:	80 3d c1 0c 20 00 00 	cmpb   $0x0,0x200cc1(%rip)        # 202318 <completed.7389>
    1657:	75 2f                	jne    1688 <__do_global_dtors_aux+0x38>
    1659:	55                   	push   %rbp
    165a:	48 83 3d 6e 09 20 00 	cmpq   $0x0,0x20096e(%rip)        # 201fd0 <__cxa_finalize@GLIBC_2.2.5>
    1661:	00 
    1662:	48 89 e5             	mov    %rsp,%rbp
    1665:	74 0c                	je     1673 <__do_global_dtors_aux+0x23>
    1667:	48 8b 3d 52 0a 20 00 	mov    0x200a52(%rip),%rdi        # 2020c0 <__dso_handle>
    166e:	e8 0d f8 ff ff       	callq  e80 <__cxa_finalize@plt>
    1673:	e8 68 ff ff ff       	callq  15e0 <deregister_tm_clones>
    1678:	c6 05 99 0c 20 00 01 	movb   $0x1,0x200c99(%rip)        # 202318 <completed.7389>
    167f:	5d                   	pop    %rbp
    1680:	c3                   	retq   
    1681:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1688:	c3                   	retq   
    1689:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001690 <frame_dummy>:
    1690:	e9 7b ff ff ff       	jmpq   1610 <register_tm_clones>
    1695:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    169c:	00 00 00 
    169f:	90                   	nop

00000000000016a0 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.4>:
    16a0:	41 57                	push   %r15
    16a2:	49 89 f7             	mov    %rsi,%r15
    16a5:	41 56                	push   %r14
    16a7:	41 55                	push   %r13
    16a9:	41 54                	push   %r12
    16ab:	55                   	push   %rbp
    16ac:	53                   	push   %rbx
    16ad:	48 83 ec 18          	sub    $0x18,%rsp
    16b1:	4c 63 6e 04          	movslq 0x4(%rsi),%r13
    16b5:	49 81 fd fc ff ff 7f 	cmp    $0x7ffffffc,%r13
    16bc:	0f 87 8e 00 00 00    	ja     1750 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.4+0xb0>
    16c2:	4d 8d 45 01          	lea    0x1(%r13),%r8
    16c6:	b8 fd ff ff 7f       	mov    $0x7ffffffd,%eax
    16cb:	31 d2                	xor    %edx,%edx
    16cd:	48 8b 0d 54 0c 20 00 	mov    0x200c54(%rip),%rcx        # 202328 <_ZL9generator>
    16d4:	49 f7 f0             	div    %r8
    16d7:	49 ba 05 00 00 00 02 	movabs $0x200000005,%r10
    16de:	00 00 00 
    16e1:	4c 0f af c0          	imul   %rax,%r8
    16e5:	49 89 c1             	mov    %rax,%r9
    16e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    16ef:	00 
    16f0:	48 69 f9 a7 41 00 00 	imul   $0x41a7,%rcx,%rdi
    16f7:	48 89 f8             	mov    %rdi,%rax
    16fa:	48 89 f9             	mov    %rdi,%rcx
    16fd:	49 f7 e2             	mul    %r10
    1700:	48 29 d1             	sub    %rdx,%rcx
    1703:	48 d1 e9             	shr    %rcx
    1706:	48 01 ca             	add    %rcx,%rdx
    1709:	48 c1 ea 1e          	shr    $0x1e,%rdx
    170d:	48 89 d6             	mov    %rdx,%rsi
    1710:	48 c1 e6 1f          	shl    $0x1f,%rsi
    1714:	48 29 d6             	sub    %rdx,%rsi
    1717:	48 29 f7             	sub    %rsi,%rdi
    171a:	48 8d 47 ff          	lea    -0x1(%rdi),%rax
    171e:	48 89 f9             	mov    %rdi,%rcx
    1721:	49 39 c0             	cmp    %rax,%r8
    1724:	76 ca                	jbe    16f0 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.4+0x50>
    1726:	31 d2                	xor    %edx,%edx
    1728:	48 89 3d f9 0b 20 00 	mov    %rdi,0x200bf9(%rip)        # 202328 <_ZL9generator>
    172f:	49 f7 f1             	div    %r9
    1732:	41 03 07             	add    (%r15),%eax
    1735:	48 83 c4 18          	add    $0x18,%rsp
    1739:	5b                   	pop    %rbx
    173a:	5d                   	pop    %rbp
    173b:	41 5c                	pop    %r12
    173d:	41 5d                	pop    %r13
    173f:	41 5e                	pop    %r14
    1741:	41 5f                	pop    %r15
    1743:	c3                   	retq   
    1744:	66 90                	xchg   %ax,%ax
    1746:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    174d:	00 00 00 
    1750:	49 81 fd fd ff ff 7f 	cmp    $0x7ffffffd,%r13
    1757:	0f 84 a3 00 00 00    	je     1800 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.4+0x160>
    175d:	4c 89 ea             	mov    %r13,%rdx
    1760:	49 89 fc             	mov    %rdi,%r12
    1763:	48 bb 09 00 00 00 02 	movabs $0x8000000200000009,%rbx
    176a:	00 00 80 
    176d:	48 d1 ea             	shr    %rdx
    1770:	4c 8d 74 24 08       	lea    0x8(%rsp),%r14
    1775:	48 8d 2d ac 0b 20 00 	lea    0x200bac(%rip),%rbp        # 202328 <_ZL9generator>
    177c:	48 89 d0             	mov    %rdx,%rax
    177f:	48 f7 e3             	mul    %rbx
    1782:	48 89 d3             	mov    %rdx,%rbx
    1785:	48 c1 eb 1d          	shr    $0x1d,%rbx
    1789:	4c 89 f6             	mov    %r14,%rsi
    178c:	4c 89 e7             	mov    %r12,%rdi
    178f:	c7 44 24 08 00 00 00 	movl   $0x0,0x8(%rsp)
    1796:	00 
    1797:	89 5c 24 0c          	mov    %ebx,0xc(%rsp)
    179b:	e8 00 ff ff ff       	callq  16a0 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.4>
    17a0:	48 69 7d 00 a7 41 00 	imul   $0x41a7,0x0(%rbp),%rdi
    17a7:	00 
    17a8:	89 c6                	mov    %eax,%esi
    17aa:	48 b8 05 00 00 00 02 	movabs $0x200000005,%rax
    17b1:	00 00 00 
    17b4:	48 f7 e7             	mul    %rdi
    17b7:	48 89 f8             	mov    %rdi,%rax
    17ba:	48 29 d0             	sub    %rdx,%rax
    17bd:	48 d1 e8             	shr    %rax
    17c0:	48 01 c2             	add    %rax,%rdx
    17c3:	48 c1 ea 1e          	shr    $0x1e,%rdx
    17c7:	48 89 d0             	mov    %rdx,%rax
    17ca:	48 c1 e0 1f          	shl    $0x1f,%rax
    17ce:	48 29 d0             	sub    %rdx,%rax
    17d1:	48 29 c7             	sub    %rax,%rdi
    17d4:	48 63 c6             	movslq %esi,%rax
    17d7:	48 69 c0 fe ff ff 7f 	imul   $0x7ffffffe,%rax,%rax
    17de:	48 89 fa             	mov    %rdi,%rdx
    17e1:	48 89 7d 00          	mov    %rdi,0x0(%rbp)
    17e5:	48 ff ca             	dec    %rdx
    17e8:	48 01 d0             	add    %rdx,%rax
    17eb:	0f 92 c2             	setb   %dl
    17ee:	0f b6 d2             	movzbl %dl,%edx
    17f1:	49 39 c5             	cmp    %rax,%r13
    17f4:	72 93                	jb     1789 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.4+0xe9>
    17f6:	48 85 d2             	test   %rdx,%rdx
    17f9:	75 8e                	jne    1789 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.4+0xe9>
    17fb:	e9 32 ff ff ff       	jmpq   1732 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.4+0x92>
    1800:	48 69 0d 1d 0b 20 00 	imul   $0x41a7,0x200b1d(%rip),%rcx        # 202328 <_ZL9generator>
    1807:	a7 41 00 00 
    180b:	48 ba 05 00 00 00 02 	movabs $0x200000005,%rdx
    1812:	00 00 00 
    1815:	48 89 c8             	mov    %rcx,%rax
    1818:	48 f7 e2             	mul    %rdx
    181b:	48 89 c8             	mov    %rcx,%rax
    181e:	48 29 d0             	sub    %rdx,%rax
    1821:	48 d1 e8             	shr    %rax
    1824:	48 01 d0             	add    %rdx,%rax
    1827:	48 c1 e8 1e          	shr    $0x1e,%rax
    182b:	48 89 c2             	mov    %rax,%rdx
    182e:	48 c1 e2 1f          	shl    $0x1f,%rdx
    1832:	48 29 c2             	sub    %rax,%rdx
    1835:	48 29 d1             	sub    %rdx,%rcx
    1838:	48 89 c8             	mov    %rcx,%rax
    183b:	48 89 0d e6 0a 20 00 	mov    %rcx,0x200ae6(%rip)        # 202328 <_ZL9generator>
    1842:	48 ff c8             	dec    %rax
    1845:	e9 e8 fe ff ff       	jmpq   1732 <_ZNSt24uniform_int_distributionIiEclISt26linear_congruential_engineImLm16807ELm0ELm2147483647EEEEiRT_RKNS0_10param_typeE.constprop.4+0x92>
    184a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001850 <__libc_csu_init>:
    1850:	41 57                	push   %r15
    1852:	41 56                	push   %r14
    1854:	49 89 d7             	mov    %rdx,%r15
    1857:	41 55                	push   %r13
    1859:	41 54                	push   %r12
    185b:	4c 8d 25 3e 05 20 00 	lea    0x20053e(%rip),%r12        # 201da0 <__frame_dummy_init_array_entry>
    1862:	55                   	push   %rbp
    1863:	48 8d 2d 4e 05 20 00 	lea    0x20054e(%rip),%rbp        # 201db8 <__init_array_end>
    186a:	53                   	push   %rbx
    186b:	41 89 fd             	mov    %edi,%r13d
    186e:	49 89 f6             	mov    %rsi,%r14
    1871:	4c 29 e5             	sub    %r12,%rbp
    1874:	48 83 ec 08          	sub    $0x8,%rsp
    1878:	48 c1 fd 03          	sar    $0x3,%rbp
    187c:	e8 97 f4 ff ff       	callq  d18 <_init>
    1881:	48 85 ed             	test   %rbp,%rbp
    1884:	74 20                	je     18a6 <__libc_csu_init+0x56>
    1886:	31 db                	xor    %ebx,%ebx
    1888:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    188f:	00 
    1890:	4c 89 fa             	mov    %r15,%rdx
    1893:	4c 89 f6             	mov    %r14,%rsi
    1896:	44 89 ef             	mov    %r13d,%edi
    1899:	41 ff 14 dc          	callq  *(%r12,%rbx,8)
    189d:	48 83 c3 01          	add    $0x1,%rbx
    18a1:	48 39 dd             	cmp    %rbx,%rbp
    18a4:	75 ea                	jne    1890 <__libc_csu_init+0x40>
    18a6:	48 83 c4 08          	add    $0x8,%rsp
    18aa:	5b                   	pop    %rbx
    18ab:	5d                   	pop    %rbp
    18ac:	41 5c                	pop    %r12
    18ae:	41 5d                	pop    %r13
    18b0:	41 5e                	pop    %r14
    18b2:	41 5f                	pop    %r15
    18b4:	c3                   	retq   
    18b5:	90                   	nop
    18b6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    18bd:	00 00 00 

00000000000018c0 <__libc_csu_fini>:
    18c0:	f3 c3                	repz retq 

Disassembly of section .fini:

00000000000018c4 <_fini>:
    18c4:	48 83 ec 08          	sub    $0x8,%rsp
    18c8:	48 83 c4 08          	add    $0x8,%rsp
    18cc:	c3                   	retq   

Disassembly of section .rodata:

00000000000018e0 <_IO_stdin_used>:
    18e0:	01 00                	add    %eax,(%rax)
    18e2:	02 00                	add    (%rax),%al
    18e4:	73 65                	jae    194b <_IO_stdin_used+0x6b>
    18e6:	20 61 63             	and    %ah,0x63(%rcx)
    18e9:	61                   	(bad)  
    18ea:	62                   	(bad)  
    18eb:	6f                   	outsl  %ds:(%rsi),(%dx)
    18ec:	20 65 6c             	and    %ah,0x6c(%rbp)
    18ef:	20 74 69 65          	and    %dh,0x65(%rcx,%rbp,2)
    18f3:	6d                   	insl   (%dx),%es:(%rdi)
    18f4:	70 6f                	jo     1965 <_IO_stdin_used+0x85>
    18f6:	0a 0a                	or     (%rdx),%cl
    18f8:	00 6c 61 20          	add    %ch,0x20(%rcx,%riz,2)
    18fc:	61                   	(bad)  
    18fd:	63 74 69 76          	movslq 0x76(%rcx,%rbp,2),%esi
    1901:	69 64 61 64 20 64 65 	imul   $0x63656420,0x64(%rcx,%riz,2),%esp
    1908:	63 
    1909:	61                   	(bad)  
    190a:	79 6f                	jns    197b <_IO_stdin_used+0x9b>
    190c:	20 61 20             	and    %ah,0x20(%rcx)
    190f:	63 65 72             	movslq 0x72(%rbp),%esp
    1912:	6f                   	outsl  %ds:(%rsi),(%dx)
    1913:	0a 0a                	or     (%rdx),%cl
    1915:	00 4c 49 53          	add    %cl,0x53(%rcx,%rcx,2)
    1919:	54                   	push   %rsp
    191a:	4f 0a 00             	rex.WRXB or (%r8),%r8b
    191d:	61                   	(bad)  
    191e:	63 74 69 76          	movslq 0x76(%rcx,%rbp,2),%esi
    1922:	69 74 79 2e 64 61 74 	imul   $0x746164,0x2e(%rcx,%rdi,2),%esi
    1929:	00 
    192a:	4c                   	rex.WR
    192b:	49 53                	rex.WB push %r11
    192d:	54                   	push   %rsp
    192e:	4f 3a 20             	rex.WRXB cmp (%r8),%r12b
    1931:	00 00                	add    %al,(%rax)
    1933:	00 00                	add    %al,(%rax)
    1935:	00 00                	add    %al,(%rax)
    1937:	00 65 73             	add    %ah,0x73(%rbp)
    193a:	74 61                	je     199d <_IO_stdin_used+0xbd>
    193c:	64 6f                	outsl  %fs:(%rsi),(%dx)
    193e:	20 69 6e             	and    %ch,0x6e(%rcx)
    1941:	69 63 69 61 6c 20 65 	imul   $0x65206c61,0x69(%rbx),%esp
    1948:	73 74                	jae    19be <_IO_stdin_used+0xde>
    194a:	61                   	(bad)  
    194b:	62                   	(bad)  
    194c:	6c                   	insb   (%dx),%es:(%rdi)
    194d:	65 20 64 65 20       	and    %ah,%gs:0x20(%rbp,%riz,2)
    1952:	6c                   	insb   (%dx),%es:(%rdi)
    1953:	61                   	(bad)  
    1954:	20 70 69             	and    %dh,0x69(%rax)
    1957:	6c                   	insb   (%dx),%es:(%rdi)
    1958:	61                   	(bad)  
    1959:	20 64 65 20          	and    %ah,0x20(%rbp,%riz,2)
    195d:	61                   	(bad)  
    195e:	72 65                	jb     19c5 <_IO_stdin_used+0xe5>
    1960:	6e                   	outsb  %ds:(%rsi),(%dx)
    1961:	61                   	(bad)  
    1962:	2e 2e 2e 00 00       	cs cs add %al,%cs:(%rax)
    1967:	00 65 73             	add    %ah,0x73(%rbp)
    196a:	74 61                	je     19cd <_IO_stdin_used+0xed>
    196c:	64 6f                	outsl  %fs:(%rsi),(%dx)
    196e:	20 69 6e             	and    %ch,0x6e(%rcx)
    1971:	69 63 69 61 6c 20 64 	imul   $0x64206c61,0x69(%rbx),%esp
    1978:	65 73 65             	gs jae 19e0 <_IO_stdin_used+0x100>
    197b:	73 74                	jae    19f1 <_IO_stdin_used+0x111>
    197d:	61                   	(bad)  
    197e:	62                   	(bad)  
    197f:	69 6c 69 7a 61 64 6f 	imul   $0x206f6461,0x7a(%rcx,%rbp,2),%ebp
    1986:	20 
    1987:	64 65 20 6c 61 20    	fs and %ch,%gs:0x20(%rcx,%riz,2)
    198d:	70 69                	jo     19f8 <_IO_stdin_used+0x118>
    198f:	6c                   	insb   (%dx),%es:(%rdi)
    1990:	61                   	(bad)  
    1991:	20 64 65 20          	and    %ah,0x20(%rbp,%riz,2)
    1995:	61                   	(bad)  
    1996:	72 65                	jb     19fd <_IO_stdin_used+0x11d>
    1998:	6e                   	outsb  %ds:(%rsi),(%dx)
    1999:	61                   	(bad)  
    199a:	2e 2e 2e 00 00       	cs cs add %al,%cs:(%rax)
    199f:	00 65 76             	add    %ah,0x76(%rbp)
    19a2:	6f                   	outsl  %ds:(%rsi),(%dx)
    19a3:	6c                   	insb   (%dx),%es:(%rdi)
    19a4:	75 63                	jne    1a09 <_IO_stdin_used+0x129>
    19a6:	69 6f 6e 20 64 65 20 	imul   $0x20656420,0x6e(%rdi),%ebp
    19ad:	6c                   	insb   (%dx),%es:(%rdi)
    19ae:	61                   	(bad)  
    19af:	20 70 69             	and    %dh,0x69(%rax)
    19b2:	6c                   	insb   (%dx),%es:(%rdi)
    19b3:	61                   	(bad)  
    19b4:	20 64 65 20          	and    %ah,0x20(%rbp,%riz,2)
    19b8:	61                   	(bad)  
    19b9:	72 65                	jb     1a20 <_IO_stdin_used+0x140>
    19bb:	6e                   	outsb  %ds:(%rsi),(%dx)
    19bc:	61                   	(bad)  
    19bd:	2e 2e 2e 00 00       	cs cs add %al,%cs:(%rax)
	...
    19e2:	00 00                	add    %al,(%rax)
    19e4:	01 00                	add    %eax,(%rax)
    19e6:	00 00                	add    %al,(%rax)
    19e8:	02 00                	add    (%rax),%al
    19ea:	00 00                	add    %al,(%rax)
    19ec:	03 00                	add    (%rax),%eax
    19ee:	00 00                	add    %al,(%rax)
    19f0:	04 00                	add    $0x0,%al
    19f2:	00 00                	add    %al,(%rax)
    19f4:	05 00 00 00 06       	add    $0x6000000,%eax
    19f9:	00 00                	add    %al,(%rax)
    19fb:	00 07                	add    %al,(%rdi)
    19fd:	00 00                	add    %al,(%rax)
    19ff:	00 08                	add    %cl,(%rax)
    1a01:	00 00                	add    %al,(%rax)
    1a03:	00 08                	add    %cl,(%rax)
    1a05:	00 00                	add    %al,(%rax)
    1a07:	00 08                	add    %cl,(%rax)
    1a09:	00 00                	add    %al,(%rax)
    1a0b:	00 08                	add    %cl,(%rax)
    1a0d:	00 00                	add    %al,(%rax)
    1a0f:	00 08                	add    %cl,(%rax)
    1a11:	00 00                	add    %al,(%rax)
    1a13:	00 08                	add    %cl,(%rax)
    1a15:	00 00                	add    %al,(%rax)
    1a17:	00 08                	add    %cl,(%rax)
    1a19:	00 00                	add    %al,(%rax)
    1a1b:	00 08                	add    %cl,(%rax)
    1a1d:	00 00                	add    %al,(%rax)
    1a1f:	00 01                	add    %al,(%rcx)
    1a21:	00 00                	add    %al,(%rax)
    1a23:	00 01                	add    %al,(%rcx)
    1a25:	00 00                	add    %al,(%rax)
    1a27:	00 01                	add    %al,(%rcx)
    1a29:	00 00                	add    %al,(%rax)
    1a2b:	00 01                	add    %al,(%rcx)
    1a2d:	00 00                	add    %al,(%rax)
    1a2f:	00 01                	add    %al,(%rcx)
    1a31:	00 00                	add    %al,(%rax)
    1a33:	00 01                	add    %al,(%rcx)
    1a35:	00 00                	add    %al,(%rax)
    1a37:	00 01                	add    %al,(%rcx)
    1a39:	00 00                	add    %al,(%rax)
    1a3b:	00 01                	add    %al,(%rcx)
    1a3d:	00 00                	add    %al,(%rax)
    1a3f:	00 29                	add    %ch,(%rcx)
    1a41:	5c                   	pop    %rsp
    1a42:	8f c2                	pop    %rdx
    1a44:	f5                   	cmc    
    1a45:	28 ec                	sub    %ch,%ah
    1a47:	3f                   	(bad)  
    1a48:	29 5c 8f c2          	sub    %ebx,-0x3e(%rdi,%rcx,4)
    1a4c:	f5                   	cmc    
    1a4d:	28 ec                	sub    %ch,%ah
    1a4f:	3f                   	(bad)  
    1a50:	29 5c 8f c2          	sub    %ebx,-0x3e(%rdi,%rcx,4)
    1a54:	f5                   	cmc    
    1a55:	28 ec                	sub    %ch,%ah
    1a57:	3f                   	(bad)  
    1a58:	29 5c 8f c2          	sub    %ebx,-0x3e(%rdi,%rcx,4)
    1a5c:	f5                   	cmc    
    1a5d:	28 ec                	sub    %ch,%ah
    1a5f:	3f                   	(bad)  

Disassembly of section .eh_frame_hdr:

0000000000001a60 <__GNU_EH_FRAME_HDR>:
    1a60:	01 1b                	add    %ebx,(%rbx)
    1a62:	03 3b                	add    (%rbx),%edi
    1a64:	54                   	push   %rsp
    1a65:	00 00                	add    %al,(%rax)
    1a67:	00 09                	add    %cl,(%rcx)
    1a69:	00 00                	add    %al,(%rax)
    1a6b:	00 d0                	add    %dl,%al
    1a6d:	f2 ff                	repnz (bad) 
    1a6f:	ff a0 00 00 00 20    	jmpq   *0x20000000(%rax)
    1a75:	f4                   	hlt    
    1a76:	ff                   	(bad)  
    1a77:	ff c8                	dec    %eax
    1a79:	00 00                	add    %al,(%rax)
    1a7b:	00 30                	add    %dh,(%rax)
    1a7d:	f4                   	hlt    
    1a7e:	ff                   	(bad)  
    1a7f:	ff 2c 01             	ljmp   *(%rcx,%rax,1)
    1a82:	00 00                	add    %al,(%rax)
    1a84:	70 f4                	jo     1a7a <__GNU_EH_FRAME_HDR+0x1a>
    1a86:	ff                   	(bad)  
    1a87:	ff 64 01 00          	jmpq   *0x0(%rcx,%rax,1)
    1a8b:	00 30                	add    %dh,(%rax)
    1a8d:	fb                   	sti    
    1a8e:	ff                   	(bad)  
    1a8f:	ff                   	(bad)  
    1a90:	f8                   	clc    
    1a91:	01 00                	add    %eax,(%rax)
    1a93:	00 50 fb             	add    %dl,-0x5(%rax)
    1a96:	ff                   	(bad)  
    1a97:	ff 70 00             	pushq  0x0(%rax)
    1a9a:	00 00                	add    %al,(%rax)
    1a9c:	40 fc                	rex cld 
    1a9e:	ff                   	(bad)  
    1a9f:	ff e0                	jmpq   *%rax
    1aa1:	00 00                	add    %al,(%rax)
    1aa3:	00 f0                	add    %dh,%al
    1aa5:	fd                   	std    
    1aa6:	ff                   	(bad)  
    1aa7:	ff 98 01 00 00 60    	lcall  *0x60000001(%rax)
    1aad:	fe                   	(bad)  
    1aae:	ff                   	(bad)  
    1aaf:	ff e0                	jmpq   *%rax
    1ab1:	01 00                	add    %eax,(%rax)
	...

Disassembly of section .eh_frame:

0000000000001ab8 <__FRAME_END__-0x1b4>:
    1ab8:	14 00                	adc    $0x0,%al
    1aba:	00 00                	add    %al,(%rax)
    1abc:	00 00                	add    %al,(%rax)
    1abe:	00 00                	add    %al,(%rax)
    1ac0:	01 7a 52             	add    %edi,0x52(%rdx)
    1ac3:	00 01                	add    %al,(%rcx)
    1ac5:	78 10                	js     1ad7 <__GNU_EH_FRAME_HDR+0x77>
    1ac7:	01 1b                	add    %ebx,(%rbx)
    1ac9:	0c 07                	or     $0x7,%al
    1acb:	08 90 01 07 10 14    	or     %dl,0x14100701(%rax)
    1ad1:	00 00                	add    %al,(%rax)
    1ad3:	00 1c 00             	add    %bl,(%rax,%rax,1)
    1ad6:	00 00                	add    %al,(%rax)
    1ad8:	d8 fa                	fdivr  %st(2),%st
    1ada:	ff                   	(bad)  
    1adb:	ff 2b                	ljmp   *(%rbx)
	...
    1ae5:	00 00                	add    %al,(%rax)
    1ae7:	00 14 00             	add    %dl,(%rax,%rax,1)
    1aea:	00 00                	add    %al,(%rax)
    1aec:	00 00                	add    %al,(%rax)
    1aee:	00 00                	add    %al,(%rax)
    1af0:	01 7a 52             	add    %edi,0x52(%rdx)
    1af3:	00 01                	add    %al,(%rcx)
    1af5:	78 10                	js     1b07 <__GNU_EH_FRAME_HDR+0xa7>
    1af7:	01 1b                	add    %ebx,(%rbx)
    1af9:	0c 07                	or     $0x7,%al
    1afb:	08 90 01 00 00 24    	or     %dl,0x24000001(%rax)
    1b01:	00 00                	add    %al,(%rax)
    1b03:	00 1c 00             	add    %bl,(%rax,%rax,1)
    1b06:	00 00                	add    %al,(%rax)
    1b08:	28 f2                	sub    %dh,%dl
    1b0a:	ff                   	(bad)  
    1b0b:	ff 50 01             	callq  *0x1(%rax)
    1b0e:	00 00                	add    %al,(%rax)
    1b10:	00 0e                	add    %cl,(%rsi)
    1b12:	10 46 0e             	adc    %al,0xe(%rsi)
    1b15:	18 4a 0f             	sbb    %cl,0xf(%rdx)
    1b18:	0b 77 08             	or     0x8(%rdi),%esi
    1b1b:	80 00 3f             	addb   $0x3f,(%rax)
    1b1e:	1a 3b                	sbb    (%rbx),%bh
    1b20:	2a 33                	sub    (%rbx),%dh
    1b22:	24 22                	and    $0x22,%al
    1b24:	00 00                	add    %al,(%rax)
    1b26:	00 00                	add    %al,(%rax)
    1b28:	14 00                	adc    $0x0,%al
    1b2a:	00 00                	add    %al,(%rax)
    1b2c:	44 00 00             	add    %r8b,(%rax)
    1b2f:	00 50 f3             	add    %dl,-0xd(%rax)
    1b32:	ff                   	(bad)  
    1b33:	ff 08                	decl   (%rax)
	...
    1b3d:	00 00                	add    %al,(%rax)
    1b3f:	00 48 00             	add    %cl,0x0(%rax)
    1b42:	00 00                	add    %al,(%rax)
    1b44:	5c                   	pop    %rsp
    1b45:	00 00                	add    %al,(%rax)
    1b47:	00 58 fb             	add    %bl,-0x5(%rax)
    1b4a:	ff                   	(bad)  
    1b4b:	ff aa 01 00 00 00    	ljmp   *0x1(%rdx)
    1b51:	42 0e                	rex.X (bad) 
    1b53:	10 8f 02 45 0e 18    	adc    %cl,0x180e4502(%rdi)
    1b59:	8e 03                	mov    (%rbx),%es
    1b5b:	42 0e                	rex.X (bad) 
    1b5d:	20 8d 04 42 0e 28    	and    %cl,0x280e4204(%rbp)
    1b63:	8c 05 41 0e 30 86    	mov    %es,-0x79cff1bf(%rip)        # ffffffff863029aa <_end+0xffffffff8610067a>
    1b69:	06                   	(bad)  
    1b6a:	41 0e                	rex.B (bad) 
    1b6c:	38 83 07 44 0e 50    	cmp    %al,0x500e4407(%rbx)
    1b72:	02 88 0a 0e 38 41    	add    0x41380e0a(%rax),%cl
    1b78:	0e                   	(bad)  
    1b79:	30 41 0e             	xor    %al,0xe(%rcx)
    1b7c:	28 42 0e             	sub    %al,0xe(%rdx)
    1b7f:	20 42 0e             	and    %al,0xe(%rdx)
    1b82:	18 42 0e             	sbb    %al,0xe(%rdx)
    1b85:	10 42 0e             	adc    %al,0xe(%rdx)
    1b88:	08 4d 0b             	or     %cl,0xb(%rbp)
    1b8b:	00 14 00             	add    %dl,(%rax,%rax,1)
    1b8e:	00 00                	add    %al,(%rax)
    1b90:	a8 00                	test   $0x0,%al
    1b92:	00 00                	add    %al,(%rax)
    1b94:	fc                   	cld    
    1b95:	f2 ff                	repnz (bad) 
    1b97:	ff                   	(bad)  
    1b98:	3a 00                	cmp    (%rax),%al
    1b9a:	00 00                	add    %al,(%rax)
    1b9c:	00 44 0e 10          	add    %al,0x10(%rsi,%rcx,1)
    1ba0:	75 0e                	jne    1bb0 <__GNU_EH_FRAME_HDR+0x150>
    1ba2:	08 00                	or     %al,(%rax)
    1ba4:	1c 00                	sbb    $0x0,%al
    1ba6:	00 00                	add    %al,(%rax)
    1ba8:	00 00                	add    %al,(%rax)
    1baa:	00 00                	add    %al,(%rax)
    1bac:	01 7a 50             	add    %edi,0x50(%rdx)
    1baf:	4c 52                	rex.WR push %rdx
    1bb1:	00 01                	add    %al,(%rcx)
    1bb3:	78 10                	js     1bc5 <__GNU_EH_FRAME_HDR+0x165>
    1bb5:	07                   	(bad)  
    1bb6:	9b                   	fwait
    1bb7:	11 05 20 00 1b 1b    	adc    %eax,0x1b1b0020(%rip)        # 1b1b1bdd <_end+0x1afaf8ad>
    1bbd:	0c 07                	or     $0x7,%al
    1bbf:	08 90 01 00 00 30    	or     %dl,0x30000001(%rax)
    1bc5:	00 00                	add    %al,(%rax)
    1bc7:	00 24 00             	add    %ah,(%rax,%rax,1)
    1bca:	00 00                	add    %al,(%rax)
    1bcc:	04 f3                	add    $0xf3,%al
    1bce:	ff                   	(bad)  
    1bcf:	ff b7 06 00 00 04    	pushq  0x4000006(%rdi)
    1bd5:	9b                   	fwait
    1bd6:	00 00                	add    %al,(%rax)
    1bd8:	00 41 0e             	add    %al,0xe(%rcx)
    1bdb:	10 86 02 45 0d 06    	adc    %al,0x60d4502(%rsi)
    1be1:	54                   	push   %rsp
    1be2:	8f 03                	popq   (%rbx)
    1be4:	8e 04 8d 05 8c 06 83 	mov    -0x7cf973fb(,%rcx,4),%es
    1beb:	07                   	(bad)  
    1bec:	03 1c 05 0a 0c 07 08 	add    0x8070c0a(,%rax,1),%ebx
    1bf3:	4a 0b 00             	rex.WX or (%rax),%rax
    1bf6:	00 00                	add    %al,(%rax)
    1bf8:	44 00 00             	add    %r8b,(%rax)
    1bfb:	00 14 01             	add    %dl,(%rcx,%rax,1)
    1bfe:	00 00                	add    %al,(%rax)
    1c00:	50                   	push   %rax
    1c01:	fc                   	cld    
    1c02:	ff                   	(bad)  
    1c03:	ff 65 00             	jmpq   *0x0(%rbp)
    1c06:	00 00                	add    %al,(%rax)
    1c08:	00 42 0e             	add    %al,0xe(%rdx)
    1c0b:	10 8f 02 42 0e 18    	adc    %cl,0x180e4202(%rdi)
    1c11:	8e 03                	mov    (%rbx),%es
    1c13:	45 0e                	rex.RB (bad) 
    1c15:	20 8d 04 42 0e 28    	and    %cl,0x280e4204(%rbp)
    1c1b:	8c 05 48 0e 30 86    	mov    %es,-0x79cff1b8(%rip)        # ffffffff86302a69 <_end+0xffffffff86100739>
    1c21:	06                   	(bad)  
    1c22:	48 0e                	rex.W (bad) 
    1c24:	38 83 07 4d 0e 40    	cmp    %al,0x400e4d07(%rbx)
    1c2a:	72 0e                	jb     1c3a <__GNU_EH_FRAME_HDR+0x1da>
    1c2c:	38 41 0e             	cmp    %al,0xe(%rcx)
    1c2f:	30 41 0e             	xor    %al,0xe(%rcx)
    1c32:	28 42 0e             	sub    %al,0xe(%rdx)
    1c35:	20 42 0e             	and    %al,0xe(%rdx)
    1c38:	18 42 0e             	sbb    %al,0xe(%rdx)
    1c3b:	10 42 0e             	adc    %al,0xe(%rdx)
    1c3e:	08 00                	or     %al,(%rax)
    1c40:	14 00                	adc    $0x0,%al
    1c42:	00 00                	add    %al,(%rax)
    1c44:	5c                   	pop    %rsp
    1c45:	01 00                	add    %eax,(%rax)
    1c47:	00 78 fc             	add    %bh,-0x4(%rax)
    1c4a:	ff                   	(bad)  
    1c4b:	ff 02                	incl   (%rdx)
	...
    1c55:	00 00                	add    %al,(%rax)
    1c57:	00 10                	add    %dl,(%rax)
    1c59:	00 00                	add    %al,(%rax)
    1c5b:	00 74 01 00          	add    %dh,0x0(%rcx,%rax,1)
    1c5f:	00 30                	add    %dh,(%rax)
    1c61:	f9                   	stc    
    1c62:	ff                   	(bad)  
    1c63:	ff 13                	callq  *(%rbx)
    1c65:	00 00                	add    %al,(%rax)
    1c67:	00 00                	add    %al,(%rax)
    1c69:	00 00                	add    %al,(%rax)
	...

0000000000001c6c <__FRAME_END__>:
    1c6c:	00 00                	add    %al,(%rax)
	...

Disassembly of section .gcc_except_table:

0000000000001c70 <.gcc_except_table>:
    1c70:	ff                   	(bad)  
    1c71:	ff 01                	incl   (%rcx)
    1c73:	2b 1a                	sub    (%rdx),%ebx
    1c75:	05 00 00 6b 05       	add    $0x56b0000,%eax
    1c7a:	fb                   	sti    
    1c7b:	0c 00                	or     $0x0,%al
    1c7d:	8b 01                	mov    (%rcx),%eax
    1c7f:	05 bf 0c 00 9a       	add    $0x9a000cbf,%eax
    1c84:	02 ac 04 00 00 a6 09 	add    0x9a60000(%rsp,%rax,1),%ch
    1c8b:	76 80                	jbe    1c0d <__GNU_EH_FRAME_HDR+0x1ad>
    1c8d:	0d 00 ee 0a 05       	or     $0x50aee00,%eax
    1c92:	ba 0c 00 dc 0c       	mov    $0xcdc000c,%edx
    1c97:	05 00 00 f1 0c       	add    $0xcf10000,%eax
    1c9c:	32 00                	xor    (%rax),%al
	...

Disassembly of section .init_array:

0000000000201da0 <__frame_dummy_init_array_entry>:
  201da0:	90                   	nop
  201da1:	16                   	(bad)  
  201da2:	00 00                	add    %al,(%rax)
  201da4:	00 00                	add    %al,(%rax)
  201da6:	00 00                	add    %al,(%rax)
  201da8:	90                   	nop
  201da9:	0e                   	(bad)  
  201daa:	00 00                	add    %al,(%rax)
  201dac:	00 00                	add    %al,(%rax)
  201dae:	00 00                	add    %al,(%rax)
  201db0:	90                   	nop
  201db1:	15 00 00 00 00       	adc    $0x0,%eax
	...

Disassembly of section .fini_array:

0000000000201db8 <__do_global_dtors_aux_fini_array_entry>:
  201db8:	50                   	push   %rax
  201db9:	16                   	(bad)  
  201dba:	00 00                	add    %al,(%rax)
  201dbc:	00 00                	add    %al,(%rax)
	...

Disassembly of section .dynamic:

0000000000201dc0 <_DYNAMIC>:
  201dc0:	01 00                	add    %eax,(%rax)
  201dc2:	00 00                	add    %al,(%rax)
  201dc4:	00 00                	add    %al,(%rax)
  201dc6:	00 00                	add    %al,(%rax)
  201dc8:	01 00                	add    %eax,(%rax)
  201dca:	00 00                	add    %al,(%rax)
  201dcc:	00 00                	add    %al,(%rax)
  201dce:	00 00                	add    %al,(%rax)
  201dd0:	01 00                	add    %eax,(%rax)
  201dd2:	00 00                	add    %al,(%rax)
  201dd4:	00 00                	add    %al,(%rax)
  201dd6:	00 00                	add    %al,(%rax)
  201dd8:	55                   	push   %rbp
  201dd9:	00 00                	add    %al,(%rax)
  201ddb:	00 00                	add    %al,(%rax)
  201ddd:	00 00                	add    %al,(%rax)
  201ddf:	00 01                	add    %al,(%rcx)
  201de1:	00 00                	add    %al,(%rax)
  201de3:	00 00                	add    %al,(%rax)
  201de5:	00 00                	add    %al,(%rax)
  201de7:	00 5f 00             	add    %bl,0x0(%rdi)
  201dea:	00 00                	add    %al,(%rax)
  201dec:	00 00                	add    %al,(%rax)
  201dee:	00 00                	add    %al,(%rax)
  201df0:	01 00                	add    %eax,(%rax)
  201df2:	00 00                	add    %al,(%rax)
  201df4:	00 00                	add    %al,(%rax)
  201df6:	00 00                	add    %al,(%rax)
  201df8:	6d                   	insl   (%dx),%es:(%rdi)
  201df9:	00 00                	add    %al,(%rax)
  201dfb:	00 00                	add    %al,(%rax)
  201dfd:	00 00                	add    %al,(%rax)
  201dff:	00 0c 00             	add    %cl,(%rax,%rax,1)
  201e02:	00 00                	add    %al,(%rax)
  201e04:	00 00                	add    %al,(%rax)
  201e06:	00 00                	add    %al,(%rax)
  201e08:	18 0d 00 00 00 00    	sbb    %cl,0x0(%rip)        # 201e0e <_DYNAMIC+0x4e>
  201e0e:	00 00                	add    %al,(%rax)
  201e10:	0d 00 00 00 00       	or     $0x0,%eax
  201e15:	00 00                	add    %al,(%rax)
  201e17:	00 c4                	add    %al,%ah
  201e19:	18 00                	sbb    %al,(%rax)
  201e1b:	00 00                	add    %al,(%rax)
  201e1d:	00 00                	add    %al,(%rax)
  201e1f:	00 19                	add    %bl,(%rcx)
  201e21:	00 00                	add    %al,(%rax)
  201e23:	00 00                	add    %al,(%rax)
  201e25:	00 00                	add    %al,(%rax)
  201e27:	00 a0 1d 20 00 00    	add    %ah,0x201d(%rax)
  201e2d:	00 00                	add    %al,(%rax)
  201e2f:	00 1b                	add    %bl,(%rbx)
  201e31:	00 00                	add    %al,(%rax)
  201e33:	00 00                	add    %al,(%rax)
  201e35:	00 00                	add    %al,(%rax)
  201e37:	00 18                	add    %bl,(%rax)
  201e39:	00 00                	add    %al,(%rax)
  201e3b:	00 00                	add    %al,(%rax)
  201e3d:	00 00                	add    %al,(%rax)
  201e3f:	00 1a                	add    %bl,(%rdx)
  201e41:	00 00                	add    %al,(%rax)
  201e43:	00 00                	add    %al,(%rax)
  201e45:	00 00                	add    %al,(%rax)
  201e47:	00 b8 1d 20 00 00    	add    %bh,0x201d(%rax)
  201e4d:	00 00                	add    %al,(%rax)
  201e4f:	00 1c 00             	add    %bl,(%rax,%rax,1)
  201e52:	00 00                	add    %al,(%rax)
  201e54:	00 00                	add    %al,(%rax)
  201e56:	00 00                	add    %al,(%rax)
  201e58:	08 00                	or     %al,(%rax)
  201e5a:	00 00                	add    %al,(%rax)
  201e5c:	00 00                	add    %al,(%rax)
  201e5e:	00 00                	add    %al,(%rax)
  201e60:	f5                   	cmc    
  201e61:	fe                   	(bad)  
  201e62:	ff 6f 00             	ljmp   *0x0(%rdi)
  201e65:	00 00                	add    %al,(%rax)
  201e67:	00 98 02 00 00 00    	add    %bl,0x2(%rax)
  201e6d:	00 00                	add    %al,(%rax)
  201e6f:	00 05 00 00 00 00    	add    %al,0x0(%rip)        # 201e75 <_DYNAMIC+0xb5>
  201e75:	00 00                	add    %al,(%rax)
  201e77:	00 90 05 00 00 00    	add    %dl,0x5(%rax)
  201e7d:	00 00                	add    %al,(%rax)
  201e7f:	00 06                	add    %al,(%rsi)
  201e81:	00 00                	add    %al,(%rax)
  201e83:	00 00                	add    %al,(%rax)
  201e85:	00 00                	add    %al,(%rax)
  201e87:	00 c0                	add    %al,%al
  201e89:	02 00                	add    (%rax),%al
  201e8b:	00 00                	add    %al,(%rax)
  201e8d:	00 00                	add    %al,(%rax)
  201e8f:	00 0a                	add    %cl,(%rdx)
  201e91:	00 00                	add    %al,(%rax)
  201e93:	00 00                	add    %al,(%rax)
  201e95:	00 00                	add    %al,(%rax)
  201e97:	00 68 03             	add    %ch,0x3(%rax)
  201e9a:	00 00                	add    %al,(%rax)
  201e9c:	00 00                	add    %al,(%rax)
  201e9e:	00 00                	add    %al,(%rax)
  201ea0:	0b 00                	or     (%rax),%eax
  201ea2:	00 00                	add    %al,(%rax)
  201ea4:	00 00                	add    %al,(%rax)
  201ea6:	00 00                	add    %al,(%rax)
  201ea8:	18 00                	sbb    %al,(%rax)
  201eaa:	00 00                	add    %al,(%rax)
  201eac:	00 00                	add    %al,(%rax)
  201eae:	00 00                	add    %al,(%rax)
  201eb0:	15 00 00 00 00       	adc    $0x0,%eax
	...
  201ebd:	00 00                	add    %al,(%rax)
  201ebf:	00 03                	add    %al,(%rbx)
	...
  201ec9:	20 20                	and    %ah,(%rax)
  201ecb:	00 00                	add    %al,(%rax)
  201ecd:	00 00                	add    %al,(%rax)
  201ecf:	00 02                	add    %al,(%rdx)
  201ed1:	00 00                	add    %al,(%rax)
  201ed3:	00 00                	add    %al,(%rax)
  201ed5:	00 00                	add    %al,(%rax)
  201ed7:	00 e0                	add    %ah,%al
  201ed9:	01 00                	add    %eax,(%rax)
  201edb:	00 00                	add    %al,(%rax)
  201edd:	00 00                	add    %al,(%rax)
  201edf:	00 14 00             	add    %dl,(%rax,%rax,1)
  201ee2:	00 00                	add    %al,(%rax)
  201ee4:	00 00                	add    %al,(%rax)
  201ee6:	00 00                	add    %al,(%rax)
  201ee8:	07                   	(bad)  
  201ee9:	00 00                	add    %al,(%rax)
  201eeb:	00 00                	add    %al,(%rax)
  201eed:	00 00                	add    %al,(%rax)
  201eef:	00 17                	add    %dl,(%rdi)
  201ef1:	00 00                	add    %al,(%rax)
  201ef3:	00 00                	add    %al,(%rax)
  201ef5:	00 00                	add    %al,(%rax)
  201ef7:	00 38                	add    %bh,(%rax)
  201ef9:	0b 00                	or     (%rax),%eax
  201efb:	00 00                	add    %al,(%rax)
  201efd:	00 00                	add    %al,(%rax)
  201eff:	00 07                	add    %al,(%rdi)
  201f01:	00 00                	add    %al,(%rax)
  201f03:	00 00                	add    %al,(%rax)
  201f05:	00 00                	add    %al,(%rax)
  201f07:	00 e8                	add    %ch,%al
  201f09:	09 00                	or     %eax,(%rax)
  201f0b:	00 00                	add    %al,(%rax)
  201f0d:	00 00                	add    %al,(%rax)
  201f0f:	00 08                	add    %cl,(%rax)
  201f11:	00 00                	add    %al,(%rax)
  201f13:	00 00                	add    %al,(%rax)
  201f15:	00 00                	add    %al,(%rax)
  201f17:	00 50 01             	add    %dl,0x1(%rax)
  201f1a:	00 00                	add    %al,(%rax)
  201f1c:	00 00                	add    %al,(%rax)
  201f1e:	00 00                	add    %al,(%rax)
  201f20:	09 00                	or     %eax,(%rax)
  201f22:	00 00                	add    %al,(%rax)
  201f24:	00 00                	add    %al,(%rax)
  201f26:	00 00                	add    %al,(%rax)
  201f28:	18 00                	sbb    %al,(%rax)
  201f2a:	00 00                	add    %al,(%rax)
  201f2c:	00 00                	add    %al,(%rax)
  201f2e:	00 00                	add    %al,(%rax)
  201f30:	fb                   	sti    
  201f31:	ff                   	(bad)  
  201f32:	ff 6f 00             	ljmp   *0x0(%rdi)
  201f35:	00 00                	add    %al,(%rax)
  201f37:	00 00                	add    %al,(%rax)
  201f39:	00 00                	add    %al,(%rax)
  201f3b:	08 00                	or     %al,(%rax)
  201f3d:	00 00                	add    %al,(%rax)
  201f3f:	00 fe                	add    %bh,%dh
  201f41:	ff                   	(bad)  
  201f42:	ff 6f 00             	ljmp   *0x0(%rdi)
  201f45:	00 00                	add    %al,(%rax)
  201f47:	00 38                	add    %bh,(%rax)
  201f49:	09 00                	or     %eax,(%rax)
  201f4b:	00 00                	add    %al,(%rax)
  201f4d:	00 00                	add    %al,(%rax)
  201f4f:	00 ff                	add    %bh,%bh
  201f51:	ff                   	(bad)  
  201f52:	ff 6f 00             	ljmp   *0x0(%rdi)
  201f55:	00 00                	add    %al,(%rax)
  201f57:	00 03                	add    %al,(%rbx)
  201f59:	00 00                	add    %al,(%rax)
  201f5b:	00 00                	add    %al,(%rax)
  201f5d:	00 00                	add    %al,(%rax)
  201f5f:	00 f0                	add    %dh,%al
  201f61:	ff                   	(bad)  
  201f62:	ff 6f 00             	ljmp   *0x0(%rdi)
  201f65:	00 00                	add    %al,(%rax)
  201f67:	00 f8                	add    %bh,%al
  201f69:	08 00                	or     %al,(%rax)
  201f6b:	00 00                	add    %al,(%rax)
  201f6d:	00 00                	add    %al,(%rax)
  201f6f:	00 f9                	add    %bh,%cl
  201f71:	ff                   	(bad)  
  201f72:	ff 6f 00             	ljmp   *0x0(%rdi)
  201f75:	00 00                	add    %al,(%rax)
  201f77:	00 05 00 00 00 00    	add    %al,0x0(%rip)        # 201f7d <_DYNAMIC+0x1bd>
	...

Disassembly of section .got:

0000000000201fd0 <.got>:
	...

Disassembly of section .got.plt:

0000000000202000 <_GLOBAL_OFFSET_TABLE_>:
  202000:	c0 1d 20 00 00 00 00 	rcrb   $0x0,0x20(%rip)        # 202027 <_GLOBAL_OFFSET_TABLE_+0x27>
	...
  202017:	00 46 0d             	add    %al,0xd(%rsi)
  20201a:	00 00                	add    %al,(%rax)
  20201c:	00 00                	add    %al,(%rax)
  20201e:	00 00                	add    %al,(%rax)
  202020:	56                   	push   %rsi
  202021:	0d 00 00 00 00       	or     $0x0,%eax
  202026:	00 00                	add    %al,(%rax)
  202028:	66 0d 00 00          	or     $0x0,%ax
  20202c:	00 00                	add    %al,(%rax)
  20202e:	00 00                	add    %al,(%rax)
  202030:	76 0d                	jbe    20203f <_GLOBAL_OFFSET_TABLE_+0x3f>
  202032:	00 00                	add    %al,(%rax)
  202034:	00 00                	add    %al,(%rax)
  202036:	00 00                	add    %al,(%rax)
  202038:	86 0d 00 00 00 00    	xchg   %cl,0x0(%rip)        # 20203e <_GLOBAL_OFFSET_TABLE_+0x3e>
  20203e:	00 00                	add    %al,(%rax)
  202040:	96                   	xchg   %eax,%esi
  202041:	0d 00 00 00 00       	or     $0x0,%eax
  202046:	00 00                	add    %al,(%rax)
  202048:	a6                   	cmpsb  %es:(%rdi),%ds:(%rsi)
  202049:	0d 00 00 00 00       	or     $0x0,%eax
  20204e:	00 00                	add    %al,(%rax)
  202050:	b6 0d                	mov    $0xd,%dh
  202052:	00 00                	add    %al,(%rax)
  202054:	00 00                	add    %al,(%rax)
  202056:	00 00                	add    %al,(%rax)
  202058:	c6                   	(bad)  
  202059:	0d 00 00 00 00       	or     $0x0,%eax
  20205e:	00 00                	add    %al,(%rax)
  202060:	d6                   	(bad)  
  202061:	0d 00 00 00 00       	or     $0x0,%eax
  202066:	00 00                	add    %al,(%rax)
  202068:	e6 0d                	out    %al,$0xd
  20206a:	00 00                	add    %al,(%rax)
  20206c:	00 00                	add    %al,(%rax)
  20206e:	00 00                	add    %al,(%rax)
  202070:	f6 0d 00 00 00 00 00 	testb  $0x0,0x0(%rip)        # 202077 <_GLOBAL_OFFSET_TABLE_+0x77>
  202077:	00 06                	add    %al,(%rsi)
  202079:	0e                   	(bad)  
  20207a:	00 00                	add    %al,(%rax)
  20207c:	00 00                	add    %al,(%rax)
  20207e:	00 00                	add    %al,(%rax)
  202080:	16                   	(bad)  
  202081:	0e                   	(bad)  
  202082:	00 00                	add    %al,(%rax)
  202084:	00 00                	add    %al,(%rax)
  202086:	00 00                	add    %al,(%rax)
  202088:	26 0e                	es (bad) 
  20208a:	00 00                	add    %al,(%rax)
  20208c:	00 00                	add    %al,(%rax)
  20208e:	00 00                	add    %al,(%rax)
  202090:	36 0e                	ss (bad) 
  202092:	00 00                	add    %al,(%rax)
  202094:	00 00                	add    %al,(%rax)
  202096:	00 00                	add    %al,(%rax)
  202098:	46 0e                	rex.RX (bad) 
  20209a:	00 00                	add    %al,(%rax)
  20209c:	00 00                	add    %al,(%rax)
  20209e:	00 00                	add    %al,(%rax)
  2020a0:	56                   	push   %rsi
  2020a1:	0e                   	(bad)  
  2020a2:	00 00                	add    %al,(%rax)
  2020a4:	00 00                	add    %al,(%rax)
  2020a6:	00 00                	add    %al,(%rax)
  2020a8:	66 0e                	data16 (bad) 
  2020aa:	00 00                	add    %al,(%rax)
  2020ac:	00 00                	add    %al,(%rax)
  2020ae:	00 00                	add    %al,(%rax)
  2020b0:	76 0e                	jbe    2020c0 <__dso_handle>
  2020b2:	00 00                	add    %al,(%rax)
  2020b4:	00 00                	add    %al,(%rax)
	...

Disassembly of section .data:

00000000002020b8 <__data_start>:
	...

00000000002020c0 <__dso_handle>:
  2020c0:	c0 20 20             	shlb   $0x20,(%rax)
  2020c3:	00 00                	add    %al,(%rax)
  2020c5:	00 00                	add    %al,(%rax)
	...

00000000002020c8 <DW.ref.__gxx_personality_v0>:
	...

Disassembly of section .bss:

00000000002020e0 <_ZSt4cout@@GLIBCXX_3.4>:
	...

0000000000202200 <_ZSt3cin@@GLIBCXX_3.4>:
	...

0000000000202318 <completed.7389>:
	...

0000000000202320 <_ZStL8__ioinit>:
	...

0000000000202328 <_ZL9generator>:
	...

Disassembly of section .comment:

0000000000000000 <.comment>:
   0:	47                   	rex.RXB
   1:	43                   	rex.XB
   2:	43 3a 20             	rex.XB cmp (%r8),%spl
   5:	28 44 65 62          	sub    %al,0x62(%rbp,%riz,2)
   9:	69 61 6e 20 38 2e 31 	imul   $0x312e3820,0x6e(%rcx),%esp
  10:	2e 30 2d 38 29 20 38 	xor    %ch,%cs:0x38202938(%rip)        # 3820294f <_end+0x3800061f>
  17:	2e 31 2e             	xor    %ebp,%cs:(%rsi)
  1a:	30 00                	xor    %al,(%rax)

Disassembly of section .debug_aranges:

0000000000000000 <.debug_aranges>:
   0:	2c 00                	sub    $0x0,%al
   2:	00 00                	add    %al,(%rax)
   4:	02 00                	add    (%rax),%al
   6:	00 00                	add    %al,(%rax)
   8:	00 00                	add    %al,(%rax)
   a:	08 00                	or     %al,(%rax)
   c:	00 00                	add    %al,(%rax)
   e:	00 00                	add    %al,(%rax)
  10:	90                   	nop
  11:	15 00 00 00 00       	adc    $0x0,%eax
  16:	00 00                	add    %al,(%rax)
  18:	13 00                	adc    (%rax),%eax
	...

Disassembly of section .debug_info:

0000000000000000 <.debug_info>:
   0:	60                   	(bad)  
   1:	00 00                	add    %al,(%rax)
   3:	00 04 00             	add    %al,(%rax,%rax,1)
   6:	00 00                	add    %al,(%rax)
   8:	00 00                	add    %al,(%rax)
   a:	08 01                	or     %al,(%rcx)
   c:	00 00                	add    %al,(%rax)
   e:	00 00                	add    %al,(%rax)
  10:	0c 7e                	or     $0x7e,%al
  12:	00 00                	add    %al,(%rax)
  14:	00 cd                	add    %cl,%ch
	...
  26:	00 00                	add    %al,(%rax)
  28:	00 02                	add    %al,(%rdx)
  2a:	b2 00                	mov    $0x0,%dl
  2c:	00 00                	add    %al,(%rax)
  2e:	01 53 01             	add    %edx,0x1(%rbx)
  31:	90                   	nop
  32:	15 00 00 00 00       	adc    $0x0,%eax
  37:	00 00                	add    %al,(%rax)
  39:	13 00                	adc    (%rax),%eax
  3b:	00 00                	add    %al,(%rax)
  3d:	00 00                	add    %al,(%rax)
  3f:	00 00                	add    %al,(%rax)
  41:	01 9c 5c 00 00 00 03 	add    %ebx,0x3000000(%rsp,%rbx,2)
  48:	ac                   	lods   %ds:(%rsi),%al
  49:	00 00                	add    %al,(%rax)
  4b:	00 01                	add    %al,(%rcx)
  4d:	5e                   	pop    %rsi
  4e:	10 5c 00 00          	adc    %bl,0x0(%rax,%rax,1)
  52:	00 06                	add    %al,(%rsi)
	...
  5c:	04 04                	add    $0x4,%al
  5e:	07                   	(bad)  
  5f:	c0 00 00             	rolb   $0x0,(%rax)
	...

Disassembly of section .debug_abbrev:

0000000000000000 <.debug_abbrev>:
   0:	01 11                	add    %edx,(%rcx)
   2:	01 25 0e 13 0b 03    	add    %esp,0x30b130e(%rip)        # 30b1316 <_end+0x2eaefe6>
   8:	0e                   	(bad)  
   9:	1b 0e                	sbb    (%rsi),%ecx
   b:	55                   	push   %rbp
   c:	17                   	(bad)  
   d:	11 01                	adc    %eax,(%rcx)
   f:	10 17                	adc    %dl,(%rdi)
  11:	00 00                	add    %al,(%rax)
  13:	02 2e                	add    (%rsi),%ch
  15:	01 03                	add    %eax,(%rbx)
  17:	0e                   	(bad)  
  18:	3a 0b                	cmp    (%rbx),%cl
  1a:	3b 0b                	cmp    (%rbx),%ecx
  1c:	39 0b                	cmp    %ecx,(%rbx)
  1e:	27                   	(bad)  
  1f:	19 11                	sbb    %edx,(%rcx)
  21:	01 12                	add    %edx,(%rdx)
  23:	07                   	(bad)  
  24:	40 18 97 42 19 01 13 	sbb    %dl,0x13011942(%rdi)
  2b:	00 00                	add    %al,(%rax)
  2d:	03 34 00             	add    (%rax,%rax,1),%esi
  30:	03 0e                	add    (%rsi),%ecx
  32:	3a 0b                	cmp    (%rbx),%cl
  34:	3b 0b                	cmp    (%rbx),%ecx
  36:	39 0b                	cmp    %ecx,(%rbx)
  38:	49 13 02             	adc    (%r10),%rax
  3b:	17                   	(bad)  
  3c:	b7 42                	mov    $0x42,%bh
  3e:	17                   	(bad)  
  3f:	00 00                	add    %al,(%rax)
  41:	04 24                	add    $0x24,%al
  43:	00 0b                	add    %cl,(%rbx)
  45:	0b 3e                	or     (%rsi),%edi
  47:	0b 03                	or     (%rbx),%eax
  49:	0e                   	(bad)  
  4a:	00 00                	add    %al,(%rax)
	...

Disassembly of section .debug_line:

0000000000000000 <.debug_line>:
   0:	73 00                	jae    2 <_init-0xd16>
   2:	00 00                	add    %al,(%rax)
   4:	02 00                	add    (%rax),%al
   6:	44 00 00             	add    %r8b,(%rax)
   9:	00 01                	add    %al,(%rcx)
   b:	01 fb                	add    %edi,%ebx
   d:	0e                   	(bad)  
   e:	0d 00 01 01 01       	or     $0x1010100,%eax
  13:	01 00                	add    %eax,(%rax)
  15:	00 00                	add    %al,(%rax)
  17:	01 00                	add    %eax,(%rax)
  19:	00 01                	add    %al,(%rcx)
  1b:	2e 2e 2f             	cs cs (bad) 
  1e:	2e 2e 2f             	cs cs (bad) 
  21:	2e 2e 2f             	cs cs (bad) 
  24:	73 72                	jae    98 <_init-0xc80>
  26:	63 2f                	movslq (%rdi),%ebp
  28:	6c                   	insb   (%dx),%es:(%rdi)
  29:	69 62 67 63 63 2f 63 	imul   $0x632f6363,0x67(%rdx),%esp
  30:	6f                   	outsl  %ds:(%rsi),(%dx)
  31:	6e                   	outsb  %ds:(%rsi),(%dx)
  32:	66 69 67 2f 69 33    	imul   $0x3369,0x2f(%rdi),%sp
  38:	38 36                	cmp    %dh,(%rsi)
  3a:	00 00                	add    %al,(%rax)
  3c:	63 72 74             	movslq 0x74(%rdx),%esi
  3f:	66 61                	data16 (bad) 
  41:	73 74                	jae    b7 <_init-0xc61>
  43:	6d                   	insl   (%dx),%es:(%rdi)
  44:	61                   	(bad)  
  45:	74 68                	je     af <_init-0xc69>
  47:	2e 63 00             	movslq %cs:(%rax),%eax
  4a:	01 00                	add    %eax,(%rax)
  4c:	00 00                	add    %al,(%rax)
  4e:	05 01 00 09 02       	add    $0x2090001,%eax
  53:	90                   	nop
  54:	15 00 00 00 00       	adc    $0x0,%eax
  59:	00 00                	add    %al,(%rax)
  5b:	03 d3                	add    %ebx,%edx
  5d:	00 01                	add    %al,(%rcx)
  5f:	05 03 03 0a 01       	add    $0x10a0303,%eax
  64:	05 18 06 01 05       	add    $0x5010618,%eax
  69:	03 06                	add    (%rsi),%eax
  6b:	59                   	pop    %rcx
  6c:	13 06                	adc    (%rsi),%eax
  6e:	82                   	(bad)  
  6f:	05 01 5a 02 01       	add    $0x1025a01,%eax
  74:	00 01                	add    %al,(%rcx)
  76:	01                   	.byte 0x1

Disassembly of section .debug_str:

0000000000000000 <.debug_str>:
   0:	47                   	rex.RXB
   1:	4e 55                	rex.WRX push %rbp
   3:	20 43 31             	and    %al,0x31(%rbx)
   6:	37                   	(bad)  
   7:	20 38                	and    %bh,(%rax)
   9:	2e 31 2e             	xor    %ebp,%cs:(%rsi)
   c:	30 20                	xor    %ah,(%rax)
   e:	2d 6d 6c 6f 6e       	sub    $0x6e6f6c6d,%eax
  13:	67 2d 64 6f 75 62    	addr32 sub $0x62756f64,%eax
  19:	6c                   	insb   (%dx),%es:(%rdi)
  1a:	65 2d 38 30 20 2d    	gs sub $0x2d203038,%eax
  20:	6d                   	insl   (%dx),%es:(%rdi)
  21:	74 75                	je     98 <_init-0xc80>
  23:	6e                   	outsb  %ds:(%rsi),(%dx)
  24:	65 3d 67 65 6e 65    	gs cmp $0x656e6567,%eax
  2a:	72 69                	jb     95 <_init-0xc83>
  2c:	63 20                	movslq (%rax),%esp
  2e:	2d 6d 61 72 63       	sub    $0x6372616d,%eax
  33:	68 3d 78 38 36       	pushq  $0x3638783d
  38:	2d 36 34 20 2d       	sub    $0x2d203436,%eax
  3d:	67 20 2d 67 20 2d 67 	and    %ch,0x672d2067(%eip)        # 672d20ab <_end+0x670cfd7b>
  44:	20 2d 4f 32 20 2d    	and    %ch,0x2d20324f(%rip)        # 2d203299 <_end+0x2d000f69>
  4a:	4f 32 20             	rex.WRXB xor (%r8),%r12b
  4d:	2d 4f 32 20 2d       	sub    $0x2d20324f,%eax
  52:	66 62                	data16 (bad) 
  54:	75 69                	jne    bf <_init-0xc59>
  56:	6c                   	insb   (%dx),%es:(%rdi)
  57:	64 69 6e 67 2d 6c 69 	imul   $0x62696c2d,%fs:0x67(%rsi),%ebp
  5e:	62 
  5f:	67 63 63 20          	movslq 0x20(%ebx),%esp
  63:	2d 66 6e 6f 2d       	sub    $0x2d6f6e66,%eax
  68:	73 74                	jae    de <_init-0xc3a>
  6a:	61                   	(bad)  
  6b:	63 6b 2d             	movslq 0x2d(%rbx),%ebp
  6e:	70 72                	jo     e2 <_init-0xc36>
  70:	6f                   	outsl  %ds:(%rsi),(%dx)
  71:	74 65                	je     d8 <_init-0xc40>
  73:	63 74 6f 72          	movslq 0x72(%rdi,%rbp,2),%esi
  77:	20 2d 66 70 69 63    	and    %ch,0x63697066(%rip)        # 636970e3 <_end+0x63494db3>
  7d:	00 2e                	add    %ch,(%rsi)
  7f:	2e 2f                	cs (bad) 
  81:	2e 2e 2f             	cs cs (bad) 
  84:	2e 2e 2f             	cs cs (bad) 
  87:	73 72                	jae    fb <_init-0xc1d>
  89:	63 2f                	movslq (%rdi),%ebp
  8b:	6c                   	insb   (%dx),%es:(%rdi)
  8c:	69 62 67 63 63 2f 63 	imul   $0x632f6363,0x67(%rdx),%esp
  93:	6f                   	outsl  %ds:(%rsi),(%dx)
  94:	6e                   	outsb  %ds:(%rsi),(%dx)
  95:	66 69 67 2f 69 33    	imul   $0x3369,0x2f(%rdi),%sp
  9b:	38 36                	cmp    %dh,(%rsi)
  9d:	2f                   	(bad)  
  9e:	63 72 74             	movslq 0x74(%rdx),%esi
  a1:	66 61                	data16 (bad) 
  a3:	73 74                	jae    119 <_init-0xbff>
  a5:	6d                   	insl   (%dx),%es:(%rdi)
  a6:	61                   	(bad)  
  a7:	74 68                	je     111 <_init-0xc07>
  a9:	2e 63 00             	movslq %cs:(%rax),%eax
  ac:	6d                   	insl   (%dx),%es:(%rdi)
  ad:	78 63                	js     112 <_init-0xc06>
  af:	73 72                	jae    123 <_init-0xbf5>
  b1:	00 73 65             	add    %dh,0x65(%rbx)
  b4:	74 5f                	je     115 <_init-0xc03>
  b6:	66 61                	data16 (bad) 
  b8:	73 74                	jae    12e <_init-0xbea>
  ba:	5f                   	pop    %rdi
  bb:	6d                   	insl   (%dx),%es:(%rdi)
  bc:	61                   	(bad)  
  bd:	74 68                	je     127 <_init-0xbf1>
  bf:	00 75 6e             	add    %dh,0x6e(%rbp)
  c2:	73 69                	jae    12d <_init-0xbeb>
  c4:	67 6e                	outsb  %ds:(%esi),(%dx)
  c6:	65 64 20 69 6e       	gs and %ch,%fs:0x6e(%rcx)
  cb:	74 00                	je     cd <_init-0xc4b>
  cd:	2f                   	(bad)  
  ce:	62                   	(bad)  
  cf:	75 69                	jne    13a <_init-0xbde>
  d1:	6c                   	insb   (%dx),%es:(%rdi)
  d2:	64 2f                	fs (bad) 
  d4:	67 63 63 2d          	movslq 0x2d(%ebx),%esp
  d8:	38 2d 75 69 53 6d    	cmp    %ch,0x6d536975(%rip)        # 6d536a53 <_end+0x6d334723>
  de:	68 68 2f 67 63       	pushq  $0x63672f68
  e3:	63 2d 38 2d 38 2e    	movslq 0x2e382d38(%rip),%ebp        # 2e382e21 <_end+0x2e180af1>
  e9:	31 2e                	xor    %ebp,(%rsi)
  eb:	30 2f                	xor    %ch,(%rdi)
  ed:	62                   	(bad)  
  ee:	75 69                	jne    159 <_init-0xbbf>
  f0:	6c                   	insb   (%dx),%es:(%rdi)
  f1:	64 2f                	fs (bad) 
  f3:	78 38                	js     12d <_init-0xbeb>
  f5:	36 5f                	ss pop %rdi
  f7:	36 34 2d             	ss xor $0x2d,%al
  fa:	6c                   	insb   (%dx),%es:(%rdi)
  fb:	69 6e 75 78 2d 67 6e 	imul   $0x6e672d78,0x75(%rsi),%ebp
 102:	75 2f                	jne    133 <_init-0xbe5>
 104:	6c                   	insb   (%dx),%es:(%rdi)
 105:	69                   	.byte 0x69
 106:	62                   	(bad)  
 107:	67 63 63 00          	movslq 0x0(%ebx),%esp

Disassembly of section .debug_loc:

0000000000000000 <.debug_loc>:
   0:	00 01                	add    %al,(%rcx)
   2:	01 00                	add    %eax,(%rax)
   4:	00 00                	add    %al,(%rax)
   6:	95                   	xchg   %eax,%ebp
   7:	15 00 00 00 00       	adc    $0x0,%eax
   c:	00 00                	add    %al,(%rax)
   e:	95                   	xchg   %eax,%ebp
   f:	15 00 00 00 00       	adc    $0x0,%eax
  14:	00 00                	add    %al,(%rax)
  16:	02 00                	add    (%rax),%al
  18:	91                   	xchg   %eax,%ecx
  19:	74 95                	je     ffffffffffffffb0 <_end+0xffffffffffdfdc80>
  1b:	15 00 00 00 00       	adc    $0x0,%eax
  20:	00 00                	add    %al,(%rax)
  22:	9d                   	popfq  
  23:	15 00 00 00 00       	adc    $0x0,%eax
  28:	00 00                	add    %al,(%rax)
  2a:	09 00                	or     %eax,(%rax)
  2c:	91                   	xchg   %eax,%ecx
  2d:	74 94                	je     ffffffffffffffc3 <_end+0xffffffffffdfdc93>
  2f:	04 0a                	add    $0xa,%al
  31:	40 80 21 9f          	rex andb $0x9f,(%rcx)
  35:	9d                   	popfq  
  36:	15 00 00 00 00       	adc    $0x0,%eax
  3b:	00 00                	add    %al,(%rax)
  3d:	a3 15 00 00 00 00 00 	movabs %eax,0x200000000000015
  44:	00 02 
  46:	00 91 74 00 00 00    	add    %dl,0x74(%rcx)
	...

Disassembly of section .debug_ranges:

0000000000000000 <.debug_ranges>:
   0:	90                   	nop
   1:	15 00 00 00 00       	adc    $0x0,%eax
   6:	00 00                	add    %al,(%rax)
   8:	a3 15 00 00 00 00 00 	movabs %eax,0x15
   f:	00 00 
	...
