#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <John Massaru Yang>
Disciplina: Computação Gráfica
Data: <03/09/2024>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    model_matrix = np.identity(4)  # Inicialização padrão
    view_matrix = np.identity(4)   # Inicialização padrão
    projection_matrix = np.identity(4)  # Inicialização padrão
    matrix_stack = []  # para matrizes de transformação
    supersample_factor = 2  # Fator de superamostragem (2x2)
    z_buffer = None  # Buffer de profundidade
    supersample_framebuffer = None  # Framebuffer para supersampling


    @staticmethod
    def convert_color(color):
        """Converte uma cor de formato [r, g, b] em [int(r*255), int(g*255), int(b*255)]."""
        if isinstance(color, list) and len(color) == 3:
            return [int(c * 255) for c in color]
        else:
            print("Cor inválida recebida. Usando cor padrão.")
            return [255, 255, 255]  # Cor padrão (branco)

    @staticmethod
    def get_colors(appearance):
        """Método de apoio para recuperar cores de um nó Appearance."""
        colors = {
            "diffuseColor": [0.8, 0.8, 0.8],  # Valor padrão
            "emissiveColor": [0.0, 0.0, 0.0],  # Valor padrão
            "specularColor": [0.0, 0.0, 0.0],  # Valor padrão
            "shininess": 0.2,  # Valor padrão
            "transparency": 0.0  # Valor padrão
        }
        if appearance and 'material' in appearance:
            material = appearance['material']
            colorPerVertex = material.get('colorPerVertex', False)
            print(f"Cor por vértice ativada? {colorPerVertex}")
            if 'diffuseColor' in material:
                colors['diffuseColor'] = material['diffuseColor']
            if 'emissiveColor' in material:
                colors['emissiveColor'] = material['emissiveColor']
            if 'specularColor' in material:
                colors['specularColor'] = material['specularColor']
            if 'shininess' in material:
                colors['shininess'] = material['shininess']
            if 'transparency' in material:
                colors['transparency'] = material['transparency']
        # Adicione o print para verificar o conteúdo do dicionário
        print(f"Conteúdo do dicionário de cores: {colors}")
        return colors
        
    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definir parâmetros para câmera, plano de corte, e framebuffer para supersampling."""
        GL.width = width * GL.supersample_factor  # Aumenta a resolução para superamostragem
        GL.height = height * GL.supersample_factor
        GL.near = near
        GL.far = far
        GL.z_buffer = np.full((GL.height, GL.width), np.inf)  # Inicializa o Z-buffer
        GL.supersample_framebuffer = np.zeros((GL.height, GL.width, 3), dtype=np.uint8)  # Framebuffer
        
    @staticmethod
    def finalize_render():
        """Aplica o downsampling após o supersampling."""
        for y in range(0, GL.height, GL.supersample_factor):
            for x in range(0, GL.width, GL.supersample_factor):
                # Calcula a cor média do bloco 2x2 ou 4x4
                color = np.mean(GL.supersample_framebuffer[y:y+GL.supersample_factor, x:x+GL.supersample_factor], axis=(0, 1))
                # Redimensiona para a resolução final
                gpu.GPU.draw_pixel([x//GL.supersample_factor, y//GL.supersample_factor], gpu.GPU.RGB8, color.astype(int))
                
    @staticmethod
    def blend_colors(background_color, foreground_color, transparency):
        """Mistura a cor do objeto com o fundo, de acordo com a transparência."""
        return (1 - transparency) * np.array(foreground_color) + transparency * np.array(background_color)

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        
        # Exemplo:
        pos_x = GL.width//2
        pos_y = GL.height//2
        gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 0])  # altera pixel
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        print("Polyline2D : lineSegments = {0}".format(lineSegments)) # imprime no terminal
        print("Polyline2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
        pos_x = GL.width//2
        pos_y = GL.height//2
        gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
        pos_x = GL.width//2
        pos_y = GL.height//2
        gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)


    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        print("TriangleSet2D : vertices = {0}".format(vertices)) # imprime no terminal
        print("TriangleSet2D : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo:
        gpu.GPU.draw_pixel([6, 8], gpu.GPU.RGB8, [255, 255, 0])  # altera pixel (u, v, tipo, r, g, b)


    @staticmethod
    def triangleSet(vertices=None, point=None, appearance=None, colors=None):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        
        """Função usada para renderizar TriangleSet com interpolação de cores e Z-buffer."""
        # Se o parâmetro point for passado, ele será usado como vertices
        if point is not None:
            vertices = point
        
        # Se as cores não forem passadas, obtemos as cores padrão do material
        if colors is None:
            colors = GL.get_colors(appearance)
        
        final_color = GL.convert_color(colors["diffuseColor"])
        transparency = colors.get('transparency', 0.0)  # Se houver transparência, é usado, senão assume-se 0
        
        triangles_drawn = set()  # Set para rastrear os triângulos já desenhados
        transformed_vertices = []  # Lista para armazenar os vértices transformados
    
        # Loop para transformar os vértices
        for i in range(0, len(vertices), 3):
            # Coordenada homogênea com w=1.0
            vertex_h = np.array([vertices[i], vertices[i + 1], vertices[i + 2], 1.0])
            
            # Aplicar a transformação do modelo
            transformed_vertex = np.matmul(GL.model_matrix, vertex_h)
            # Aplicar a transformação da câmera (view)
            transformed_vertex = np.matmul(GL.view_matrix, transformed_vertex)
            # Aplicar a projeção em perspectiva
            transformed_vertex = np.matmul(GL.projection_matrix, transformed_vertex)
    
            # Divisão por w para normalizar as coordenadas de perspectiva
            if transformed_vertex[3] != 0:
                transformed_vertex /= transformed_vertex[3]
    
            # Converter para coordenadas de tela
            y = int((-transformed_vertex[1] + 1) * GL.height / 2)
            x = int((transformed_vertex[0] + 1) * GL.width / 2)
    
            # Verificar se as coordenadas estão dentro dos limites da tela
            if 0 <= x < GL.width and 0 <= y < GL.height:
                # Adicionar os vértices transformados à lista
                transformed_vertices.append([x, y, transformed_vertex[2]])  # z permanece o mesmo
        
        # Verificação para garantir que há pelo menos 3 vértices para formar um triângulo
        if len(transformed_vertices) >= 3:
            # Loop para desenhar os triângulos
            for i in range(0, len(transformed_vertices), 3):
                v1 = transformed_vertices[i]
                v2 = transformed_vertices[i + 1]
                v3 = transformed_vertices[i + 2]
    
                # Criação de uma chave única para identificar o triângulo
                triangle_key = tuple(sorted([tuple(v1), tuple(v2), tuple(v3)]))
    
                # Verificar se o triângulo já foi desenhado
                if triangle_key not in triangles_drawn:
                    # Se não foi desenhado, desenhamos o triângulo
                    print(f"Desenhando triângulo com vértices transformados: {v1}, {v2}, {v3}")
                    GL.draw_triangle(v1, v2, v3, final_color, final_color, final_color, transparency=transparency)
                    triangles_drawn.add(triangle_key)  # Adicionar o triângulo ao set de triângulos desenhados
                else:
                    print(f"Triângulo {triangle_key} já desenhado, ignorando.")


    @staticmethod
    def draw_triangle(v1, v2, v3, color1, color2, color3, colorPerVertex= True,  transparency=0.0):
        """Rasteriza um triângulo na tela."""
        # Se as cores dos vértices forem iguais, usar uma cor única para o triângulo
        if color2 is None: color2 = color1
        if color3 is None: color3 = color1
        
        # Força cores diferentes se colorPerVertex for True
        if colorPerVertex:
            color1 = [255, 0, 0]  # Vermelho
            color2 = [0, 255, 0]  # Verde
            color3 = [0, 0, 255]  # Azul
    
        def edge_interpolate(y, x0, y0, x1, y1):
            if y1 == y0:
                return x0
            return x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    
        # Ordena os vértices por y (v1.y <= v2.y <= v3.y)
        if v1[1] > v2[1]:
            v1, v2 = v2, v1
            color1, color2 = color2, color1
        if v2[1] > v3[1]:
            v2, v3 = v3, v2
            color2, color3 = color3, color2
        if v1[1] > v2[1]:
            v1, v2 = v2, v1
            color1, color2 = color2, color1
    
        # Rasterização da parte superior do triângulo
        for y in range(int(v1[1]), int(v2[1]) + 1):
            x_start = edge_interpolate(y, v1[0], v1[1], v3[0], v3[1])
            x_end = edge_interpolate(y, v1[0], v1[1], v2[0], v2[1])
            if x_start > x_end:
                x_start, x_end = x_end, x_start

            for x in range(int(x_start), int(x_end) + 1):
                if 0 <= x < GL.width and 0 <= y < GL.height:
                    if GL.z_buffer[y, x] > v1[2]:  # Verifica o Z-buffer
                        alpha, beta, gamma = GL.calculate_baricentric_coords([x, y], v1, v2, v3)
                        interpolated_color = (
                            alpha * np.array(color1) +
                            beta * np.array(color2) +
                            gamma * np.array(color3)
                        )
                        interpolated_color = np.clip(interpolated_color, 0, 255).astype(int)

                        GL.z_buffer[y, x] = v1[2]
                        background_color = GL.supersample_framebuffer[y, x]
                        blended_color = GL.blend_colors(background_color, interpolated_color, transparency)
                        GL.supersample_framebuffer[y, x] = blended_color.astype(int)

        # Rasterização da parte inferior do triângulo
        for y in range(int(v2[1]), int(v3[1]) + 1):
            x_start = edge_interpolate(y, v1[0], v1[1], v3[0], v3[1])
            x_end = edge_interpolate(y, v2[0], v2[1], v3[0], v3[1])
            if x_start > x_end:
                x_start, x_end = x_end, x_start

            for x in range(int(x_start), int(x_end) + 1):
                if 0 <= x < GL.width and 0 <= y < GL.height:
                    if GL.z_buffer[y, x] > v1[2]:  # Verifica o Z-buffer
                        alpha, beta, gamma = GL.calculate_baricentric_coords([x, y], v1, v2, v3)
                        interpolated_color = (
                            alpha * np.array(color1) +
                            beta * np.array(color2) +
                            gamma * np.array(color3)
                        )
                        interpolated_color = np.clip(interpolated_color, 0, 255).astype(int)

                        GL.z_buffer[y, x] = v1[2]
                        background_color = GL.supersample_framebuffer[y, x]
                        blended_color = GL.blend_colors(background_color, interpolated_color, transparency)
                        GL.supersample_framebuffer[y, x] = blended_color.astype(int)
                        
                    print(f"Desenhando triângulo: v1={v1}, v2={v2}, v3={v3}")
             

        @staticmethod
        def render_scene():
            points1 = [0.0, 0.5, 0.0,  -0.5, -0.5, 0.0,  0.5, -0.5, 0.0]
            points2 = [-0.5, 0.5, 0.0,  -1.0, -0.5, 0.0,  0.0, -0.5, 0.0]
            points3 = [0.5, 0.5, 0.0,  0.0, -0.5, 0.0,  1.0, -0.5, 0.0]

            """Função que decide se deve aplicar a inversão do eixo Y com base na cena."""
            GL.triangleSet(points1, {'emissiveColor': [1.0, 0.0, 0.0]})  # Triângulo vermelho
            GL.triangleSet(points2, {'emissiveColor': [0.0, 1.0, 0.0]})  # Triângulo verde
            GL.triangleSet(points3, {'emissiveColor': [0.0, 0.0, 1.0]})  # Triângulo azul
            

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

      
        eye = np.array(position)
        direction = np.array([0, 0, -1])  # Inicialmente olha para o eixo -Z
        up = np.array([0, 1, 0])  # Vetor "up" padrão
    
        # Verificar a matriz de visualização
        GL.view_matrix = GL.look_at(eye, eye + direction, up)
        print(f"Matriz de Visualização: \n{GL.view_matrix}")
        
        # Verificar a matriz de projeção
        fov = np.deg2rad(fieldOfView) if fieldOfView > np.pi else fieldOfView
        aspect_ratio = GL.width / GL.height
        near = GL.near
        far = GL.far
        GL.projection_matrix = GL.perspective_projection(fov, aspect_ratio, near, far)
        print(f"Matriz de Projeção: \n{GL.projection_matrix}")
        
        

     
    @staticmethod
    def look_at(eye, center, up):
        """Cria a matriz de visualização (LookAt)."""
        f = (center - eye) / np.linalg.norm(center - eye)
        s = np.cross(f, up) / np.linalg.norm(np.cross(f, up))
        u = np.cross(s, f)
        view_matrix = np.identity(4)
        view_matrix[0, :3] = s
        view_matrix[1, :3] = u
        view_matrix[2, :3] = -f
        view_matrix[:3, 3] = -np.dot(view_matrix[:3, :3], eye)
        return view_matrix

    @staticmethod
    def perspective_projection(fov, aspect, near, far):
        """Cria a matriz de projeção perspectiva."""
        f = 1.0 / np.tan(fov / 2)
        projection_matrix = np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ])
        print("Matriz de Projeção Perspectiva: \n", projection_matrix)
        return projection_matrix

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo em alguma estrutura de pilha.
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        scale_matrix = np.diag([scale[0], scale[1], scale[2], 1.0])
    
        # Criar matriz de rotação
        angle = rotation[3]
        axis = np.array(rotation[:3])
        rotation_matrix = GL.create_rotation_matrix(axis, angle)
        
        # Criar matriz de translação
        translation_matrix = np.identity(4)
        translation_matrix[:3, 3] = translation
        
        # Multiplicar na ordem: escala -> rotação -> translação
        GL.model_matrix = np.matmul(translation_matrix, np.matmul(rotation_matrix, scale_matrix))
        
        # Empilhar a matriz resultante
        GL.matrix_stack.append(GL.model_matrix)


        print("Transform : ", end='')
        if translation:
            print("translation = {0} ".format(translation), end='') # imprime no terminal
        if scale:
            print("scale = {0} ".format(scale), end='') # imprime no terminal
        if rotation:
            print("rotation = {0} ".format(rotation), end='') # imprime no terminal
        print("")

    @staticmethod
    def create_rotation_matrix(axis, angle):
        # Normalizar o eixo de rotação
        axis = axis / np.linalg.norm(axis)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        ux, uy, uz = axis
        
        rotation_matrix = np.array([
            [cos_angle + ux*ux*(1-cos_angle), ux*uy*(1-cos_angle) - uz*sin_angle, ux*uz*(1-cos_angle) + uy*sin_angle, 0],
            [uy*ux*(1-cos_angle) + uz*sin_angle, cos_angle + uy*uy*(1-cos_angle), uy*uz*(1-cos_angle) - ux*sin_angle, 0],
            [uz*ux*(1-cos_angle) - uy*sin_angle, uz*uy*(1-cos_angle) + ux*sin_angle, cos_angle + uz*uz*(1-cos_angle), 0],
            [0, 0, 0, 1]
        ])
        
        return rotation_matrix

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.
        if GL.matrix_stack:
            GL.model_matrix = GL.matrix_stack.pop()
        
        

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        final_color = GL.convert_color(colors['emissiveColor'])
        vertices = np.array(point).reshape(-1, 3)
        transformed_vertices = []
    
        for vertex in vertices:
            vertex_h = np.append(vertex, 1)
            transformed_vertex = np.matmul(GL.model_matrix, vertex_h)
            transformed_vertex = np.matmul(GL.view_matrix, transformed_vertex)
            transformed_vertex = np.matmul(GL.projection_matrix, transformed_vertex)
    
            if transformed_vertex[3] != 0:
                transformed_vertex /= transformed_vertex[3]
    
            x = int((transformed_vertex[0] + 1) * GL.width / 2)
            y = int((-transformed_vertex[1] + 1) * GL.height / 2)
    
            if 0 <= x < GL.width and 0 <= y < GL.height:
                transformed_vertices.append([x, y, transformed_vertex[2]])
    
        # Desenhar tiras de triângulos
        for i in range(2, len(transformed_vertices)):
            v1 = transformed_vertices[i - 2]
            v2 = transformed_vertices[i - 1]
            v3 = transformed_vertices[i]
    
            if i % 2 == 0:
                GL.draw_triangle(v1, v2, v3, final_color, final_color, final_color)
            else:
                GL.draw_triangle(v1, v3, v2, final_color, final_color, final_color)

        
        

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.
        final_color = [255, 0, 0]
        vertices = np.array(point).reshape(-1, 3)
        indexed_vertices = [vertices[i] for i in index if i != -1]
        transformed_vertices = []

        for vertex in indexed_vertices:
            vertex_h = np.append(vertex, 1)
            transformed_vertex = np.matmul(GL.model_matrix, vertex_h)
            transformed_vertex = np.matmul(GL.view_matrix, transformed_vertex)
            transformed_vertex = np.matmul(GL.projection_matrix, transformed_vertex)

            if transformed_vertex[3] != 0:
                transformed_vertex /= transformed_vertex[3]

            x = int((transformed_vertex[0] + 1) * GL.width / 2)
            y = int((-transformed_vertex[1] + 1) * GL.height / 2)

            if 0 <= x < GL.width and 0 <= y < GL.height:
                transformed_vertices.append([x, y, transformed_vertex[2]])

        for i in range(2, len(transformed_vertices)):
            v1 = transformed_vertices[i - 2]
            v2 = transformed_vertices[i - 1]
            v3 = transformed_vertices[i]

            if i % 2 == 0:
                GL.draw_triangle(v1, v2, v3, final_color)
            else:
                GL.draw_triangle(v1, v3, v2, final_color)

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        

    @staticmethod
    def calculate_baricentric_coords(p, p1, p2, p3):
        """Calcula as coordenadas baricêntricas de um ponto p dentro de um triângulo (p1, p2, p3)."""
        # Converter os pontos em arrays NumPy para que operações vetoriais funcionem
        p = np.array(p[:2])  # Pegando apenas as coordenadas x e y
        p1 = np.array(p1[:2])  # Ignorando a coordenada z
        p2 = np.array(p2[:2])
        p3 = np.array(p3[:2])
    
        # Calcular a área do triângulo no espaço 2D
        area = np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    
        # Evitar divisão por zero no cálculo de área
        if area == 0:
            return 0, 0, 0
    
        # Calcular as coordenadas baricêntricas no espaço 2D
        alpha = np.linalg.norm(np.cross(p2 - p, p3 - p)) / area
        beta = np.linalg.norm(np.cross(p3 - p, p1 - p)) / area
        gamma = 1.0 - alpha - beta
    
        return alpha, beta, gamma
    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.
        
        
        
        final_color = GL.convert_color(colors.get('emissiveColor', [1.0, 1.0, 1.0]))  # Cor padrão: branco
        vertices = np.array(coord).reshape(-1, 3)
        transformed_vertices = []

        for vertex in vertices:
            vertex_h = np.append(vertex, 1)
            transformed_vertex = np.matmul(GL.model_matrix, vertex_h)
            transformed_vertex = np.matmul(GL.view_matrix, transformed_vertex)
            transformed_vertex = np.matmul(GL.projection_matrix, transformed_vertex)

            if transformed_vertex[3] != 0:
                transformed_vertex /= transformed_vertex[3]

            x = int((transformed_vertex[0] + 1) * GL.width / 2)
            y = int((-transformed_vertex[1] + 1) * GL.height / 2)
            z = transformed_vertex[2]

            if 0 <= x < GL.width and 0 <= y < GL.height:
                transformed_vertices.append([x, y, z])

        # Processar triângulos com os índices de coordenadas e cores
        i = 0
        while i + 2 < len(coordIndex):
            if coordIndex[i] == -1 or coordIndex[i+1] == -1 or coordIndex[i+2] == -1:
                i += 1
                continue

            try:
                v1 = transformed_vertices[coordIndex[i]]
                v2 = transformed_vertices[coordIndex[i+1]]
                v3 = transformed_vertices[coordIndex[i+2]]

                if colorPerVertex and color and colorIndex:
                    try:
                        c1 = GL.convert_color(color[colorIndex[i]] if isinstance(color[colorIndex[i]], list) else final_color)
                        c2 = GL.convert_color(color[colorIndex[i+1]] if isinstance(color[colorIndex[i+1]], list) else final_color)
                        c3 = GL.convert_color(color[colorIndex[i+2]] if isinstance(color[colorIndex[i+2]], list) else final_color)
                        GL.draw_triangle(v1, v2, v3, c1, c2, c3, colorPerVertex=True)
                    except IndexError:
                        print("Erro: índice de cor não encontrado ou inválido. Usando cor padrão.")
                        GL.draw_triangle(v1, v2, v3, final_color, final_color, final_color)
                else:
                    GL.draw_triangle(v1, v2, v3, final_color,final_color, final_color)
            except IndexError:
                print(f"Erro de índice: {coordIndex[i]}, {coordIndex[i+1]}, {coordIndex[i+2]}")
            i += 3
        
            

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

       

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        
    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        
    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed


        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.


        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""