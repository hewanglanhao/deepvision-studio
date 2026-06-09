package com.deepvision.studio.forward;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.Locale;
import java.util.UUID;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
public class LocalImageStorage {
  private final Path uploadsRoot;

  /** 保存上传根目录，并规范化绝对路径，后续写入预览图时会防止路径逃逸。 */
  public LocalImageStorage(@Value("${deepvision.uploads.root}") String uploadsRoot) {
    this.uploadsRoot = Path.of(uploadsRoot).toAbsolutePath().normalize();
  }

  /** 将前端生成的输入预览 Data URL 解码落盘，返回可被历史记录列表访问的相对路径。 */
  public String saveDataUrl(Long userId, String dataUrl) {
    if (dataUrl == null || dataUrl.isBlank()) {
      return null;
    }
    int comma = dataUrl.indexOf(',');
    if (!dataUrl.startsWith("data:image/") || comma < 0) {
      throw new IllegalArgumentException("Preview image must be a data URL.");
    }

    String header = dataUrl.substring(0, comma).toLowerCase(Locale.ROOT);
    String extension = header.contains("image/jpeg") || header.contains("image/jpg") ? "jpg" : "png";
    byte[] bytes = Base64.getDecoder().decode(dataUrl.substring(comma + 1));
    if (bytes.length > 8 * 1024 * 1024) {
      throw new IllegalArgumentException("Preview image is too large.");
    }

    try {
      Path directory = uploadsRoot.resolve("a-records").resolve(String.valueOf(userId)).normalize();
      if (!directory.startsWith(uploadsRoot)) {
        throw new IllegalArgumentException("Invalid upload path.");
      }
      Files.createDirectories(directory);
      String filename = UUID.randomUUID() + "." + extension;
      Files.write(directory.resolve(filename), bytes);
      return "/uploads/a-records/" + userId + "/" + filename;
    } catch (IOException ex) {
      throw new IllegalArgumentException("Failed to save preview image.");
    }
  }
}
