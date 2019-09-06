![Image Logo](https://github.com/mittalgovind/fifty/blob/master/fifty_logo.png)
# FiFTy: Large-scale File Fragment Type Identification using Neural Networks

FiFTy is a file type classifier that works much like the ``file`` command in Unix-like systems but with much more cool techniques up its sleeve. It beats several previous benchmarks on the biggest play field there is right now.  FiFTy comes with pre-trained models for six scenarios and for block sizes of 512 and 4096 bytes.  It is retrainable for a subset of our studied filetypes and can be scaled up for newer filetypes and other block sizes too. Please find our corresponding paper at https://arxiv.org/abs/1908.06148 and the ready-to-use open access datasets at [FFT-75](https://ieee-dataport.org/open-access/file-fragment-type-fft-75-dataset).

## List of Filetypes
The classifier has been tested on the following 75 filetypes :--
| Filetype | Tag | Description |
| :----- | :-----: | :------------: |
| ARW | Raw | Raw Sony camera images |
| CR2 | Raw | Raw Canon camera images |
| DNG | Raw | Raw Adobe camera images |
| GPR | Raw | Raw GoPro camera images |
| NEF | Raw | Raw Nikon camera images |
| NRW | Raw | Raw Nikon camera images |
| ORF | Raw | Raw Olympus camera images |
| PEF | Raw | Raw Pentax camera images |
| RAF | Raw | Raw Fuji camera images |
| RW2 | Raw | Raw Panasonic camera images |
| 3FR | Raw | Raw Hasselblad camera images |
| JPG | Bitmap | Joint Photographers Experts Group (JPEG) |
| TIFF | Bitmap | Tagged Image File Format |
| HEIC | Bitmap | High Efficiency Image Format based on video frames |
| BMP | Bitmap | Bitmap images |
| GIF | Bitmap | Graphic Interchange Format |
| PNG | Bitmap | Portable Network Graphics |
| AI | Vector | Adobe Illustrator vector image |
| EPS | Vector | Encapsulated PostScript vector |
| PSD | Vector | Photoshop vector file |
| MOV | Video | QuickTime File Format |
| MP4 | Video | MPEG-4 Part 14 |
| 3GP | Video | Multimedia container videos format |
| AVI | Video | Audio Video Interleave container format |
| MKV | Video | Matroska Multimedia Container |
| OGV | Video | Ogg Vorbis video encoding format |
| WEBM | Video | Web videos |
| APK | Archive | Android application package |
| JAR | Archive | Java class package (compiled) |
| MSI | Archive | Windows Installer |
| DMG | Archive | macOS application package |
| 7Z | Archive | 7-zip archive |
| BZ2 | Archive | Burrows Wheeler archive |
| DEB | Archive | Linux/Unix application package |
| GZ | Archive | GNU Gzip |
| PKG | Archive | macOS compressed installer |
| RAR | Archive | Roshal Archive by Microsoft |
| RPM | Archive | RPM package manager (Red Hat) |
| XZ | Archive | XZ (GNU LGPL/GPL) |
| ZIP | Archive | ZIP archive |
| EXE | Executables | Windows executable |
| MACH-O | Executables | macOS executable |
| ELF | Executables | Linux executable |
| DLL | Executables | Dynamic Link Library (Windows Executable) |
| DOC | Office | Microsoft Office (2007) Word  |
| DOCX | Office | Microsoft Office (2013) Word  |
| KEY | Office | macOS keynote presentation  | converted from .pptx files|
| PPT | Office | Microsoft Office (2007) Powerpoint  |
| PPTX | Office | Microsoft Office (2013) Powerpoint  |
| XLS | Office | Microsoft Office (2007) Excel |
| XLSX | Office | Microsoft Office (2013) Excel |
| DJVU | Published | Digital Document Format by Yann LeCun |
| EPUB | Published | Electronic Publication for iBooks |
| MOBI | Published | Kindle E-book |
| PDF | Published | Portable Document Format |
| MD | Human-readable | Markdown |
| RTF | Human-readable | Rich text format |
| TXT | Human-readable | Text file |
| TEX | Human-readable | LaTeX |
| JSON | Human-readable | JavaScript Object Notation for database |
| HTML | Human-readable | HyperText Markup Language |
| XML | Human-readable | Extensible Markup Language |
| LOG | Human-readable | Log files |
| CSV | Human-readable | Comma-separated values |
| AIFF | Audio | Audio Interchange File Format |
| FLAC | Audio | Free Lossless Audio Codec |
| M4A | Audio | Audio-only MPEG-4 |
| MP3 | Audio | MPEG-1/2 Audio Layer III |
| OGG | Audio | Audio container format developed by Xiph-Org |
| WAV | Audio | Waveform Audio File format |
| WMA | Audio | Windows Media Audio developed by Microsoft |
| PCAP | Other | Wireshark captured network packets |
| TTF | Other | True-type font |
| DWG | Other | CAD drawing |
| SQLITE | Other | SQL database |


## Scenario Description
We present [models](https://github.com/mittalgovind/fifty/tree/master/fifty/utilities/models) for _six_ scenarios on two popular block sizes of __512__ and __4096__ bytes. File type selection reflects focus on media carving applications, where scenarios \#3 to \#6 are the most relevant:

1. **\#1 (All; 75 classes)**: All filetypes are separate classes; this is the most generic case and can be aggregated into more specialized use-cases.

2. **\#2 (Use-specific; 11)**: Filetypes are grouped into 11 classes according to their use; this information may be useful for more-detailed, hierarchical classification or for determining the primary use of an unknown device.

3. **\#3 (Media Carver - Photos \& Videos; 25)**: Every filetype tagged as a bitmap (6), RAW photo (11) or video (7) is considered as a separate class;  all remaining types are grouped into one _other_ class. 

4. **\#4 (Coarse Photo Carver; 5)**: Separate classes for different photographic types: JPEG, 11 raw images, 7 videos, 5 remaining bitmaps are grouped into one separate class per category; all remaining types are grouped into one _other_ class.  

5. **\#5 (Specialized JPEG Carver; 2)**: JPEG is a separate class and the remaining 74 filetypes are grouped into one _other_ class; scenario intended for analyzing disk images from generic devices.

1. **\#6 (Camera-Specialized JPEG Carver; 2)**: JPEG is a separate class and the remaining photographic/video types (11 raw images, 3GP, MOV, MKV, TIFF and HEIC) are grouped into one _other_ class; scenario intended for analyzing SD cards from digital cameras.

## Training your own file classifier

## Output run on forensic images

