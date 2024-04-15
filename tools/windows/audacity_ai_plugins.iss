; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define MyAppName "OpenVINO AI Plugins for Audacity"
#define MyAppPublisher "Intel Corporation"
#define MyAppURL "https://github.com/intel/openvino-plugins-ai-audacity/"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{944D0498-C914-46E9-97E0-813B726CB0B0}
AppName={#MyAppName}
AppVersion={#AI_PLUGIN_VERSION}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
;DefaultDirName=C:\Program Files\Audacity
;DefaultDirName={reg:HKCR\Audacity.Project,Path|C:\Program Files\Audacity}
DefaultDirName={code:FindAudacity}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DirExistsWarning=no
; Uncomment the following line to run in non administrative install mode (install for current user only.)
;PrivilegesRequired=lowest
OutputBaseFilename=mysetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
; "ArchitecturesInstallIn64BitMode=x64" requests that the install be
; done in "64-bit mode" on x64, meaning it should use the native
; 64-bit Program Files directory and the 64-bit view of the registry.
ArchitecturesInstallIn64BitMode=x64
InfoBeforeFile={#AI_PLUGIN_REPO_SOURCE_FOLDER}\LICENSE.txt
DisableWelcomePage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Types]
Name: "recommended"; Description: "Install recommended models"
Name: "all"; Description: "Install all models"
Name: "none"; Description: "Install no models"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "music_sep"; Description: "Music Separation Models"; Types: all recommended; ExtraDiskSpaceRequired: 103042178
Name: "noise_sup"; Description: "Noise Suppression Models"; Types: all
Name: "noise_sup\deepfilternet2"; Description: "DeepFilterNet2"; Types: all recommended; ExtraDiskSpaceRequired: 9700989
Name: "noise_sup\deepfilternet3"; Description: "DeepFilterNet3"; Types: all recommended; ExtraDiskSpaceRequired: 9045784
Name: "noise_sup\denseunet"; Description: "noise-suppression-denseunet-ll-0001"; Types: all; ExtraDiskSpaceRequired: 9315388
Name: "whisper"; Description: "Whisper Transcription (Speech-to-Text) Models"; Types: all
Name: "whisper\base"; Description: "Base (74M)"; Types: all recommended; ExtraDiskSpaceRequired: 189420073
Name: "whisper\small"; Description: "Small (244M)"; Types: all; ExtraDiskSpaceRequired: 664733264
Name: "whisper\small_en_tdrz"; Description: "Small(English-only) + Speaker Segmentation via tinydiarize (experimental) (244M)"; Types: all; ExtraDiskSpaceRequired: 664446765
Name: "whisper\medium"; Description: "Medium (769 M)"; Types: all; ExtraDiskSpaceRequired: 2763649009
Name: "whisper\large_v1"; Description: "Large v1 (1550 M)"; Types: all; ExtraDiskSpaceRequired: 5643116237
Name: "whisper\large_v2"; Description: "Large v2 (1550 M)"; Types: all; ExtraDiskSpaceRequired: 5643116237
Name: "whisper\large_v3"; Description: "Large v3 (1550 M)"; Types: all; ExtraDiskSpaceRequired: 5644263320  
Name: "music_gen"; Description: "Music Generation Models"; Types: all recommended; ExtraDiskSpaceRequired: 342495982
Name: "music_gen\small_mono"; Description: "Small Mono Model"; Types: all recommended; ExtraDiskSpaceRequired: 1220980259
Name: "music_gen\small_stereo"; Description: "Small Stereo Model"; Types: all; ExtraDiskSpaceRequired: 1355200757


[Files]

; OpenVINO libraries
Source: "{#OPENVINO_DIR}\runtime\bin\intel64\Release\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs
Source: "{#OPENVINO_DIR}\runtime\3rdparty\tbb\bin\*.dll"; DestDir: "{app}"; Flags: ignoreversion 
Source: "{#OPENVINO_TOKENIZERS_DIR}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

; Libtorch
Source: "{#LIBTORCH_DIR}\lib\*.dll"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs

; Whisper
Source: "{#BUILD_FOLDER}\whisper-build-avx\installed\bin\whisper.dll"; DestDir: "{app}"; Check: IsAVX2Supported();
Source: "{#BUILD_FOLDER}\whisper-build-no-avx\installed\bin\whisper.dll"; DestDir: "{app}"; Check: not IsAVX2Supported();

; Audacity module
Source: "{#BUILD_FOLDER}\audacity-build\{#AUDACITY_BUILD_CONFIG}\modules\mod-openvino.dll"; DestDir: "{app}\modules"; Flags: ignoreversion

Source: "module_preferences.bmp"; Flags: dontcopy

; NOTE: Don't use "Flags: ignoreversion" on any shared system files

Source: "{tmp}\ggml-base-models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\ggml-base-models.zip', '{app}\openvino-models'); Components: whisper\base; Flags: external
Source: "{tmp}\ggml-small-models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\ggml-small-models.zip', '{app}\openvino-models'); Components: whisper\small; Flags: external
Source: "{tmp}\ggml-small.en-tdrz-models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\ggml-small.en-tdrz-models.zip', '{app}\openvino-models'); Components: whisper\small_en_tdrz; Flags: external
Source: "{tmp}\ggml-medium-models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\ggml-medium-models.zip', '{app}\openvino-models'); Components: whisper\medium; Flags: external
Source: "{tmp}\ggml-large-v1-models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\ggml-large-v1-models.zip', '{app}\openvino-models'); Components: whisper\large_v1; Flags: external
Source: "{tmp}\ggml-large-v2-models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\ggml-large-v2-models.zip', '{app}\openvino-models'); Components: whisper\large_v2; Flags: external
Source: "{tmp}\ggml-large-v3-models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\ggml-large-v3-models.zip', '{app}\openvino-models'); Components: whisper\large_v3; Flags: external


Source: "{tmp}\musicgen_small_enc_dec_tok_openvino_models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\musicgen_small_enc_dec_tok_openvino_models.zip', '{app}\openvino-models\musicgen'); Components: music_gen; Flags: external
Source: "{tmp}\musicgen_small_stereo_openvino_models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\musicgen_small_stereo_openvino_models.zip', '{app}\openvino-models\musicgen'); Components: music_gen\small_stereo; Flags: external
Source: "{tmp}\musicgen_small_mono_openvino_models.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\musicgen_small_mono_openvino_models.zip', '{app}\openvino-models\musicgen'); Components: music_gen\small_mono; Flags: external
Source: "{tmp}\deepfilternet2.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\deepfilternet2.zip', '{app}\openvino-models\'); Components: noise_sup\deepfilternet2; Flags: external
Source: "{tmp}\deepfilternet3.zip"; DestDir: "{tmp}"; AfterInstall: ExtractSomething('{tmp}\deepfilternet3.zip', '{app}\openvino-models\'); Components: noise_sup\deepfilternet3; Flags: external
Source: "{tmp}\htdemucs_v4.bin"; DestDir: "{app}\openvino-models\";  Components: music_sep; Flags: external
Source: "{tmp}\htdemucs_v4.xml"; DestDir: "{app}\openvino-models\";  Components: music_sep; Flags: external
Source: "{tmp}\noise-suppression-denseunet-ll-0001.bin"; DestDir: "{app}\openvino-models\";  Components: noise_sup\denseunet; Flags: external
Source: "{tmp}\noise-suppression-denseunet-ll-0001.xml"; DestDir: "{app}\openvino-models\";  Components: noise_sup\denseunet; Flags: external



[Dirs]
Name: "{app}\openvino-models"
Name: "{app}\openvino-models\musicgen"; Components: music_gen;

[UninstallDelete]
Type: filesandordirs; Name: "{app}\openvino-models"
;Type: filesandordirs; Name: "{userappdata}\audacity\openvino-model-cache"

[Icons]
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"

[Code]
const
  PF_AVX2_INSTRUCTIONS_AVAILABLE = 40;

function IsProcessorFeaturePresent(ProcessorFeature: DWORD): BOOL;
  external 'IsProcessorFeaturePresent@kernel32.dll stdcall';

var
  DownloadPage: TDownloadWizardPage;
  ExtraDiskSpaceRequired: Integer;
  SupportsAVX2: Boolean;
  CustomImage: TBitmapImage;
  DocumentationLabel: TLabel;

function IsAVX2Supported(): Boolean;
begin
   Result := SupportsAVX2;
end;

function InitializeSetup(): Boolean;
begin
  SupportsAVX2 := IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE);
  Result := True;
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  if Progress = ProgressMax then
    Log(Format('Successfully downloaded file to {tmp}: %s', [FileName]));
  Result := True;
end;


procedure InitializeWizard;
begin
  // Initialize your extra disk space requirement, based on your file sizes
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing), SetupMessage(msgPreparingDesc), @OnDownloadProgress);

  WizardForm.FinishedLabel.AutoSize := True;
  WizardForm.FinishedLabel.WordWrap := True;

  CustomImage := TBitmapImage.Create(WizardForm);
  CustomImage.Parent := WizardForm.FinishedPage;
  CustomImage.AutoSize := True;

  DocumentationLabel := TLabel.Create(WizardForm);
  DocumentationLabel.Parent := WizardForm.FinishedPage;
  DocumentationLabel.AutoSize := True;
  

end;

function FindAudacity(Param: String): String;
var
  V: String;
begin
  Log('FindAudacity called');
  if RegQueryStringValue(HKCR, 'Audacity.Project\shell\open\command', '', V) then
    begin
      V := RemoveQuotes(V);
      { MsgBox('Value is "' + V + '"', mbInformation, MB_OK);}
      Result := ExtractFilePath(V);
      { MsgBox('Returning ' + Result, mbInformation, MB_OK);}

      { TODO: We should run Audacity.exe --version, grab the result, and check for particular version. }
    end
  else
    begin
       MsgBox('Location of Audacity installation could not be determined. Note that it was looking for a 64-bit installation of Audacity. Please manually set Audacity installation folder that contains Audacity.exe', mbInformation, MB_OK);
       Result := 'C:\Program Files\Audacity'
    end;
end;

function NextButtonClick(PageId: Integer): Boolean;
begin
    Result := True;
    if (PageId = wpSelectDir) and not FileExists(ExpandConstant('{app}\Audacity.exe')) then begin
        MsgBox('Audacity does not seem to be installed in that folder.  Please select a folder that contains Audacity.exe', mbError, MB_OK);
        Result := False;
        exit;
    end;
    {Make sure that all HF download links below are from specific commit-id's, not latest tip! This way, a particular installer is guaranteed to pull same version, even if repo's files change}
    {SHA256 hash can be generated from git bash terminal like this: sha256sum somefile.zip}
    if (PageId = wpSelectComponents) then 
    begin
        DownloadPage.Clear;

        if WizardIsComponentSelected('whisper\base') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/194efafbee09390bbf33f33d02d9856dacfdd162/ggml-base-models.zip?download=true', 
                             'ggml-base-models.zip', 
                             '18f6e87146bfe9986fa764657cca26695bcb7fa3ba5d77d6d6af1d38de720e4c');
        end;
        
        if WizardIsComponentSelected('whisper\small') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/194efafbee09390bbf33f33d02d9856dacfdd162/ggml-small-models.zip?download=true', 
                             'ggml-small-models.zip', 
                             '7efabdcb08ec09ee24b615db9fe86425114a46d87a25d1e3d14ecd7195dd4e9f');
        end;

        if WizardIsComponentSelected('whisper\small_en_tdrz') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/194efafbee09390bbf33f33d02d9856dacfdd162/ggml-small.en-tdrz-models.zip?download=true', 
                             'ggml-small.en-tdrz-models.zip', 
                             'ea825cb105e48aa0796e332c93827339dd60aa1a61633df14bb6611bcef26b95');
        end;

        if WizardIsComponentSelected('whisper\medium') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/5c2f20da06aa17198dbd778d4190383557c23b1b/ggml-medium-models.zip?download=true', 
                             'ggml-medium-models.zip', 
                             'afd961c0bac5830dbcbd15826f49553a5d5fb8a09c7a24eb634304fa068fe59f');
        end;

        if WizardIsComponentSelected('whisper\large_v1') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/9731937f2b9ca0aff8b8acd031ed77a1a378cc72/ggml-large-v1-models.zip?download=true', 
                             'ggml-large-v1-models.zip', 
                             'fa187861eb46b701f242d63fc0067878de6345a71714d926c4b0eecf1ec0fffa');
        end;

        if WizardIsComponentSelected('whisper\large_v2') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/dc08ae819b895594cae08b0210bee051462aba75/ggml-large-v2-models.zip?download=true', 
                             'ggml-large-v2-models.zip', 
                             'ab306b0a0a93a731e56efbfa4e80448206bd9b5b863d5a99e40a3532d64ec754');
        end;

        if WizardIsComponentSelected('whisper\large_v3') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/whisper.cpp-openvino-models/resolve/d0c3075c40e4938fb2a4cccfc91704106d3e7d12/ggml-large-v3-models.zip?download=true', 
                             'ggml-large-v3-models.zip', 
                             'd61160dc5b1c1abc1dbdd6ae571fe4da87079fa3170156408f8775b493c15cdd');
        end;

        if WizardIsComponentSelected('music_gen') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/musicgen-static-openvino/resolve/b2ad8083f3924ed704814b68c5df9cbbf2ad2aae/musicgen_small_enc_dec_tok_openvino_models.zip?download=true', 
                             'musicgen_small_enc_dec_tok_openvino_models.zip', 
                             'f5cdd695a52cf4c0619dd419c8bff4e71700ee805b547e86b7483423883d6f32');
        end;

        if WizardIsComponentSelected('music_gen\small_stereo') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/musicgen-static-openvino/resolve/b2ad8083f3924ed704814b68c5df9cbbf2ad2aae/musicgen_small_stereo_openvino_models.zip?download=true', 
                             'musicgen_small_stereo_openvino_models.zip', 
                             '1c6496267b22175ecdd2518b9ed8c9870454674fc08fac1c864ca62e3d87e8e7');
        end;

        if WizardIsComponentSelected('music_gen\small_mono') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/musicgen-static-openvino/resolve/b2ad8083f3924ed704814b68c5df9cbbf2ad2aae/musicgen_small_mono_openvino_models.zip?download=true', 
                             'musicgen_small_mono_openvino_models.zip', 
                             'b1e47df3ec60d0c5f70ba664fea636df9bc94710269ee72357a3353fd579afd9');
        end;

        if WizardIsComponentSelected('noise_sup\deepfilternet2') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/deepfilternet-openvino/resolve/995706bda3da69da0825074ba7dbc8a78067e980/deepfilternet2.zip?download=true',
                             'deepfilternet2.zip', 
                             '9d10c9c77a64121587e7d94a3d4d5cfeb67e9bfd2e416d48d1f74ac725c4f66e');
        end;

        if WizardIsComponentSelected('noise_sup\deepfilternet3') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/deepfilternet-openvino/resolve/995706bda3da69da0825074ba7dbc8a78067e980/deepfilternet3.zip?download=true',
                             'deepfilternet3.zip',  
                             'fdc74e11439ca09a106f5bd77ed24002ef4d0e02114238a271c069d06172f341');
        end;

        if WizardIsComponentSelected('noise_sup\denseunet') then 
        begin
            DownloadPage.Add('https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/noise-suppression-denseunet-ll-0001.bin',
                             'noise-suppression-denseunet-ll-0001.bin',  
                             'da59b41a656b2948a4b45580b4870614d2dfa071a7566555aea4547423888e08');

            DownloadPage.Add('https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/noise-suppression-denseunet-ll-0001/FP16/noise-suppression-denseunet-ll-0001.xml',
                             'noise-suppression-denseunet-ll-0001.xml',  
                             '89116a01cc59f7ac3f1f1365c851e47c9089fd33ca86712c30dc1f0ee7d14803');
        end;

        if WizardIsComponentSelected('music_sep') then 
        begin
            DownloadPage.Add('https://huggingface.co/Intel/demucs-openvino/resolve/97fc578fb57650045d40b00bc84c7d156be77547/htdemucs_v4.bin?download=true',
                             'htdemucs_v4.bin',  
                             '7aa84fa1f2b534bd6865a5609b8b5b028802fe761d6a09b1d30a1564f8fac6f8');

            DownloadPage.Add('https://huggingface.co/Intel/demucs-openvino/resolve/97fc578fb57650045d40b00bc84c7d156be77547/htdemucs_v4.xml?download=true',
                             'htdemucs_v4.xml',  
                             '304e24325756089d6bb6583171dd1bea2327505e87c6cb6e010afea7463d9f0a');
        end;

        DownloadPage.Show;
        try
          try
            DownloadPage.Download; // This downloads the files to {tmp}
            Result := True;
          except
            if DownloadPage.AbortedByUser then
              Log('Aborted by user.')
            else
              SuppressibleMsgBox(AddPeriod(GetExceptionMessage), mbCriticalError, MB_OK, IDOK);
            Result := False;
          end;
        finally
          DownloadPage.Hide;
        end;
        
         
    end;
end;


const
  SHCONTCH_NOPROGRESSBOX = 4;
  SHCONTCH_RESPONDYESTOALL = 16;

procedure UnZip(ZipPath, TargetPath: string); 
var
  Shell: Variant;
  ZipFile: Variant;
  TargetFolder: Variant;
begin
  Shell := CreateOleObject('Shell.Application');

  ZipFile := Shell.NameSpace(ZipPath);
  if VarIsClear(ZipFile) then
    RaiseException(
      Format('ZIP file "%s" does not exist or cannot be opened', [ZipPath]));

  TargetFolder := Shell.NameSpace(TargetPath);
  if VarIsClear(TargetFolder) then
    RaiseException(Format('Target path "%s" does not exist', [TargetPath]));

  TargetFolder.CopyHere(
    ZipFile.Items, SHCONTCH_NOPROGRESSBOX or SHCONTCH_RESPONDYESTOALL);
end;

procedure UnZipCurrentFileTo(TargetPath: string);
begin
  Log('Extracting ' + CurrentFileName + ' to ' + TargetPath)
  UnZip(ExpandConstant(CurrentFileName), ExpandConstant(TargetPath));
end;

procedure ExtractSomething(src, target: string);
begin
  UnZip(ExpandConstant(src), ExpandConstant(target));
end;


procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then
  begin
    // Modify the default finished message
    WizardForm.FinishedLabel.Font.Size := 10;
    WizardForm.FinishedLabel.Caption := 'Installation Complete. To enable use of these plugins:' + #13#10 
                                       +' 1. Open Audacity and navigate to Preferences -> Modules ' + #13#10 
                                       +' 2. Set "mod-openvino" module to "enabled"' + #13#10 
                                       +' 3. Close & then re-open Audacity.';

    

    // Load the custom image
    ExtractTemporaryFile('module_preferences.bmp');
    CustomImage.Bitmap.LoadFromFile(ExpandConstant('{tmp}\module_preferences.bmp'));

    CustomImage.Top := WizardForm.FinishedLabel.Top + WizardForm.FinishedLabel.Height + 10;
    CustomImage.Left := WizardForm.FinishedLabel.Left;

    DocumentationLabel.Caption := 'For full documentation, visit:' + #13#10 
                                 +'https://github.com/intel/openvino-plugins-ai-audacity';
    DocumentationLabel.Top := CustomImage.Top + CustomImage.Height + 30;
    DocumentationLabel.Font.Size := 10
  end;
end;



