# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# 这是处理你辅助脚本的关键部分。
# 将当前目录添加到 PyInstaller 的 Python 路径中。
# 这能确保它找到 IMUReader, GPSReader3, 和 utils。
pathex = ['.']

a = Analysis(['sensors_collect_v2.py'],
            pathex=pathex,
            binaries=[],
            datas=[],
            # 隐藏导入对于 numpy 和 pyqtgraph 等库通常是必需的
            hiddenimports=[
                'pyqtgraph.opengl.items.GLGridItem',
                'pyqtgraph.opengl.items.GLAxisItem',
                'pyqtgraph.colormap',
                'scipy.special', # Numpy 有时有隐藏依赖
                'scipy.linalg',
                'yaml',
                'rospkg',
                'catkin_pkg',
                'catkin_pkg',
                'catkin_pkg.tool',
                'defusedxml',
                'netifaces'  # Another common one for ROS
            ],
            hookspath=[],
            hooksconfig={},
            runtime_hooks=[],
            excludes=[],
            win_no_prefer_redirects=False,
            win_private_assemblies=False,
            cipher=block_cipher,
            noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
            cipher=block_cipher)

exe = EXE(pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='sensors_collect_v2',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True,  # True = 显示终端, False = 纯 GUI
        disable_windowed_traceback=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None )

coll = COLLECT(exe,
            a.binaries,
            a.zipfiles,
            a.datas,
            strip=False,
            upx=True,
            upx_exclude=[],
            name='sensors_collect_v2_dist') # 输出文件夹的名称