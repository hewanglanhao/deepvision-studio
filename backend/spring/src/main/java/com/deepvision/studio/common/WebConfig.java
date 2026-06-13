package com.deepvision.studio.common;

import java.nio.file.Path;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {
  private final Path uploadsRoot;
  private final Path datasetsRoot;

  /** 读取上传和数据集根目录，后续把用户样本图片和训练数据以静态资源方式暴露给前端。 */
  public WebConfig(
      @Value("${deepvision.uploads.root}") String uploadsRoot,
      @Value("${deepvision.datasets.root}") String datasetsRoot
  ) {
    this.uploadsRoot = Path.of(uploadsRoot).toAbsolutePath().normalize();
    this.datasetsRoot = Path.of(datasetsRoot).toAbsolutePath().normalize();
  }

  @Override
  /** 注册静态资源映射，A 模式上传的样本图和数据集预览图才能被浏览器直接加载。 */
  public void addResourceHandlers(ResourceHandlerRegistry registry) {
    registry
        .addResourceHandler("/uploads/**")
        .addResourceLocations(uploadsRoot.toUri().toString() + "/");
    registry
        .addResourceHandler("/datasets/builtin/**")
        .addResourceLocations(datasetsRoot.resolve("builtin").toUri().toString() + "/");
  }
}
