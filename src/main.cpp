#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "Fluid.hpp"

#include <array>
#include <random>

#include <SDL.h>

const int scale = 2;

void close(SDL_Window *win, SDL_Renderer *renderer) {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(win);
    SDL_Quit();
}

std::array<float, 3> HSBtoRGB(float h, float s, float b) {
    s = s / 100;
    b = b / 100;
    auto k = [h](int n) -> float { return std::fmod(n + h / 60, 6); };
    auto f = [b, s, k](int n) -> float {
        return b * (1 - s * std::max(0.0f, std::min({k(n), 4.0f - k(n), 1.0f})));
    };
    return {255 * f(5), 255 * f(3), 255 * f(1)};
}

void draw(SDL_Renderer *renderer, Fluid *fluidInstance) {
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);

    SDL_Rect r;
    r.w = scale;
    r.h = scale;

    // Draw density of dye in fluid
    auto densityMap = fluidInstance->getDensity();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float d = densityMap[i + j * n];
            auto color = HSBtoRGB(std::fmod(d + 50, 255), 200, d);
            SDL_SetRenderDrawColor(renderer, color[0], color[1], color[2], 255);
            r.x = i * scale;
            r.y = j * scale;
            SDL_RenderFillRect(renderer, &r);
        }
    }

    SDL_RenderPresent(renderer);
}

int main() {
    // get GPU
    MTL::Device *device = MTL::CreateSystemDefaultDevice();

    // Initialise fluid
    Fluid fluidInstance = Fluid(device, 30, 0, 0.0000001);

    const int width = n * scale;
    const int height = n * scale;

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *win = SDL_CreateWindow(
        "Fluid Simulation", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        width, height, SDL_WINDOW_SHOWN);
    SDL_Renderer *renderer =
        SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> densityDist(0, 50);

    std::default_random_engine angleGen;
    std::uniform_real_distribution<float> angleDist(0, 360);

    SDL_Event event;
    bool isQuit = false;
    bool isPressed = false;

    while (!isQuit) {
        if (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                isQuit = true;
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                isPressed = true;
            } else if (event.type == SDL_MOUSEBUTTONUP) {
                isPressed = false;
            }
        }

        // Choose angle and split into components of a unit vector
        float angle = angleDist(angleGen);
        float xVal = std::cos(angle * (3.141592654 / 180)) * 0.8;
        float yVal = std::sin(angle * (3.141592654 / 180)) * 0.8;

        // Add dye/density to fluid
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                fluidInstance.addDensity((n / 2) + i, (n / 2) + j,
                                         densityDist(rng));
            }
        }

        // Add velocity to fluid in direction of angle
        for (int i = 0; i < 2; i++) {
            fluidInstance.addVelocity(n / 2, n / 2, xVal, yVal);
        }

        fluidInstance.step();

        draw(renderer, &fluidInstance);
    }

    close(win, renderer);

    return 0;
}
