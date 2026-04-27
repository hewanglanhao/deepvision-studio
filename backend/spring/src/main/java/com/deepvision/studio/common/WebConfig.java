package com.deepvision.studio.common;

import java.nio.file.Path;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {
  private final Path uploadsRoot;

  public WebConfig(@Value("${deepvision.uploads.root}") String uploadsRoot) {
    this.uploadsRoot = Path.of(uploadsRoot).toAbsolutePath().normalize();
  }

  @Override
  public void addResourceHandlers(ResourceHandlerRegistry registry) {
    registry
        .addResourceHandler("/uploads/**")
        .addResourceLocations(uploadsRoot.toUri().toString() + "/");
  }
}

