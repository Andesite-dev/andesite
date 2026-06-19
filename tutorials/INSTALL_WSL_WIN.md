# **Instalar Linux (Ubuntu 24.04) en Windows mediante WSL**

1. Para instalar WSL (Subsistema de Linux para windows) se debe abrir la consola de Powershell en modo administrador y escribir:
```powershell
wsl --install
```

  > En caso que esto no resulte y no nos deje iniciar la consola de Ubuntu, seguir estos pasos:
  
  * Habilitar WSL para Linux en Powershell con permisos de administrador mediante este comando: 
  ```powershell
  dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
  ```
          
  * Habilitar Virtual Machine features en windows. Luego reiniciar el computador
  ```powershell
  dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```
        
  * Instalar el ejecutable de WSL nuevamente `wsl --install`

2. Dejar WSL 2 como defecto, escribiendo esto en powershell
```powershell
wsl --set-default-version 2
```
        
3. Instalar `Ubuntu 24.04.6` desde la aplicacion [Microsoft Store](https://apps.microsoft.com/detail/9pdxgncfsczv?hl=es-ES&gl=ES) que viene en Windows


## Problemas con instalacion

1. En caso que salga el error que se muestra a continuación, tenemos que activar **Hyper-V** en la Bios
        
```powershell
$ Please enable the Virtual Machine Platform Windows feature and ensure virtualization is enabled in the BIOS.
$ For information please visit https://aka.ms/wsl2-install
$ Press any key to continue...
```
        
> Dentro de la Bios, en `Configuration` generalmente, se encuentra la opción `Intel Virtual Technology` y dejarlo `enable`, luego de esto guardar los cambios en la BIOS y reiniciar el computador
